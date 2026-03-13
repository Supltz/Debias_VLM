from __future__ import annotations

from typing import Callable, Dict, Mapping, Sequence

import torch


TextEncoder = Callable[[list[str]], torch.Tensor]


def _as_unit_tensor(embeddings: torch.Tensor) -> torch.Tensor:
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.as_tensor(embeddings)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings [N, D], got shape={tuple(embeddings.shape)}")
    embeddings = embeddings.float()
    return torch.nn.functional.normalize(embeddings, dim=-1)


def spherical_mean(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    unit = _as_unit_tensor(embeddings)
    mean_vec = unit.mean(dim=0)
    norm = torch.linalg.norm(mean_vec)
    if torch.isnan(norm) or norm.item() <= eps:
        raise ValueError("Spherical mean is ill-defined: mean vector norm is ~0")
    return mean_vec / norm


def build_group_prototypes(
    group_prompt_map: Mapping[str, Mapping[str, object]],
    text_encoder: TextEncoder,
    sensitive_attribute: str,
) -> Dict[str, torch.Tensor]:
    prototypes: Dict[str, torch.Tensor] = {}

    for group, entry in group_prompt_map.items():
        entry_attr = str(entry.get("sensitive_attribute", "")).strip().lower()
        expected_attr = str(sensitive_attribute).strip().lower()
        if entry_attr and entry_attr != expected_attr:
            raise ValueError(
                f"Group '{group}' has sensitive_attribute='{entry_attr}', expected '{expected_attr}'"
            )

        t_g = entry.get("t_g")
        t_g_variants = entry.get("T_g", [])

        if not isinstance(t_g, str) or not t_g.strip():
            raise ValueError(f"Group '{group}' has invalid t_g")
        if t_g_variants is None:
            t_g_variants = []
        if not isinstance(t_g_variants, Sequence) or isinstance(t_g_variants, (str, bytes)):
            raise ValueError(f"Group '{group}' has invalid T_g; expected a sequence of strings")

        prompts = [t_g] + [p for p in t_g_variants if isinstance(p, str) and p.strip()]
        embeddings = text_encoder(prompts)
        prototypes[group] = spherical_mean(embeddings)

    return prototypes


def build_class_prompt_group_prototypes(
    llm_classification_map: Mapping[str, object],
    class_prompt: str,
    text_encoder: TextEncoder,
    sensitive_attribute: str,
) -> Dict[str, torch.Tensor]:
    group_prompt_map = llm_classification_map.get(class_prompt)
    if not isinstance(group_prompt_map, Mapping):
        raise KeyError(f"Class prompt not found in LLM map: {class_prompt}")
    return build_group_prototypes(
        group_prompt_map=group_prompt_map,
        text_encoder=text_encoder,
        sensitive_attribute=sensitive_attribute,
    )


def construct_attribute_space(
    prototypes: Mapping[str, torch.Tensor],
    reference_group: str | None = None,
    rank_tol: float = 1e-6,
) -> torch.Tensor:
    if len(prototypes) < 2:
        raise ValueError("Need at least two group prototypes to construct attribute space")

    group_names = list(prototypes.keys())
    ref_group = reference_group if reference_group is not None else group_names[0]
    if ref_group not in prototypes:
        raise KeyError(f"reference_group '{ref_group}' not found in prototypes")

    ref = prototypes[ref_group].float()
    if ref.ndim != 1:
        raise ValueError(f"Reference prototype must be 1D, got shape={tuple(ref.shape)}")

    diffs: list[torch.Tensor] = []
    for g in group_names:
        if g == ref_group:
            continue
        v = prototypes[g].float()
        if v.ndim != 1:
            raise ValueError(f"Prototype for group '{g}' must be 1D, got shape={tuple(v.shape)}")
        if v.shape[0] != ref.shape[0]:
            raise ValueError(f"Dimension mismatch: '{g}' has {v.shape[0]} but ref has {ref.shape[0]}")
        diffs.append(v - ref)

    D = torch.stack(diffs, dim=1)
    if D.numel() == 0:
        return D.new_zeros((ref.shape[0], 0))

    U, S, _ = torch.linalg.svd(D, full_matrices=False)
    if S.numel() == 0:
        return D.new_zeros((ref.shape[0], 0))

    s0 = torch.max(S).item() if S.numel() > 0 else 0.0
    tol = max(rank_tol * max(D.shape), 1e-12) * (s0 if s0 > 0 else 1.0)
    r = int(torch.sum(S > tol).item())
    if r == 0:
        return D.new_zeros((ref.shape[0], 0))
    A = U[:, :r]
    return A


def decompose_embedding_by_attribute_space(
    u: torch.Tensor,
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(u, torch.Tensor):
        u = torch.as_tensor(u)
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)

    u = u.float()
    A = A.float()

    squeeze_back = False
    if u.ndim == 1:
        u = u.unsqueeze(0)
        squeeze_back = True
    if u.ndim != 2:
        raise ValueError(f"u must be [d] or [m, d], got shape={tuple(u.shape)}")
    if A.ndim != 2:
        raise ValueError(f"A must be [d, r], got shape={tuple(A.shape)}")
    if u.shape[1] != A.shape[0]:
        raise ValueError(f"Dimension mismatch: u has d={u.shape[1]}, A has d={A.shape[0]}")

    if A.shape[1] == 0:
        u_parallel = torch.zeros_like(u)
        u_orth = u.clone()
    else:
        AtA = A.T @ A
        AtA_inv = torch.linalg.pinv(AtA)
        P_parallel = A @ AtA_inv @ A.T
        u_parallel = (P_parallel @ u.T).T
        u_orth = u - u_parallel

    if squeeze_back:
        return u_parallel[0], u_orth[0]
    return u_parallel, u_orth


def compute_optimal_debiased_embedding(
    e: torch.Tensor,
    A: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    if not isinstance(e, torch.Tensor):
        e = torch.as_tensor(e)
    e = e.float()

    squeeze_back = False
    if e.ndim == 1:
        e = e.unsqueeze(0)
        squeeze_back = True
    if e.ndim != 2:
        raise ValueError(f"e must be [d] or [m, d], got shape={tuple(e.shape)}")

    e_parallel, e_orth = decompose_embedding_by_attribute_space(e, A)
    n_parallel = torch.linalg.norm(e_parallel, dim=1)
    n_orth = torch.linalg.norm(e_orth, dim=1)

    n_parallel_safe = torch.clamp(n_parallel, min=eps)
    n_orth_safe = torch.clamp(n_orth, min=eps)

    e_parallel_hat = e_parallel / n_parallel_safe.unsqueeze(1)
    e_orth_hat = e_orth / n_orth_safe.unsqueeze(1)

    E = n_parallel_safe + (1.0 - n_orth) / n_parallel_safe
    radicand = torch.clamp(E * E - n_parallel * n_parallel, min=0.0)
    alpha = (E - n_orth * torch.sqrt(radicand)) / (E * E + n_orth * n_orth + eps)
    alpha = torch.clamp(alpha, min=-1.0, max=1.0)

    orth_coef = torch.sqrt(torch.clamp(1.0 - alpha * alpha, min=0.0))
    u_star = orth_coef.unsqueeze(1) * e_orth_hat + alpha.unsqueeze(1) * e_parallel_hat

    unchanged_mask = (n_parallel <= eps) | (n_orth <= eps)
    if torch.any(unchanged_mask):
        u_star[unchanged_mask] = e[unchanged_mask]

    u_star = torch.nn.functional.normalize(u_star, dim=-1)
    if squeeze_back:
        return u_star[0]
    return u_star


__all__ = [
    "spherical_mean",
    "build_group_prototypes",
    "build_class_prompt_group_prototypes",
    "construct_attribute_space",
    "decompose_embedding_by_attribute_space",
    "compute_optimal_debiased_embedding",
]

from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

# Suppress known upstream warning noise from dependency versions.
warnings.filterwarnings(
    "ignore",
    message=r"optree is installed but the version is too old to support PyTorch Dynamo in C\+\+ pytree\..*",
    category=FutureWarning,
)

import torch
import yaml
from PIL import Image
from tqdm import tqdm

from debiasing import (
    build_group_prototypes,
    compute_optimal_debiased_embedding,
    construct_attribute_space,
)
from models import load_model


DEFAULT_CONFIG_PATH = "/vol/lian/Gen_AI/config.yaml"
DEFAULT_LLM_JSON_PATH = "/vol/lian/Gen_AI/LLM.json"
DEFAULT_COCO_GROUP_CSV = "/vol/lian/Gen_AI/COCO2017.csv"
MAXSKEW_M = 1000
RETRIEVAL_SENSITIVE_ATTRIBUTE = {
    "coco": "intersection of perceived gender and skin tone",
    "flickr": "perceived gender",
}
RETRIEVAL_LLM_GROUPS = {
    "coco": ["light-skinned male", "light-skinned female", "dark-skinned male", "dark-skinned female"],
    "flickr": ["male", "female"],
}
GENDER_WORD_PATTERN = re.compile(
    r"\b(man|woman|boy|girl|gentleman|guy|lady|female|male)\b",
    flags=re.IGNORECASE,
)
COCO_SKIN_TONE_PATTERN = re.compile(
    r"\b(?:"
    r"light[- ]skinned|dark[- ]skinned|fair[- ]skinned|pale[- ]skinned|"
    r"tan[- ]skinned|brown[- ]skinned|black[- ]skinned|white[- ]skinned|"
    r"olive[- ]skinned|dusky[- ]skinned|deep[- ]skinned|medium[- ]skinned|"
    r"light complexion(?:ed)?|dark complexion(?:ed)?|fair complexion(?:ed)?|"
    r"pale complexion(?:ed)?|tan complexion(?:ed)?|brown complexion(?:ed)?|"
    r"black complexion(?:ed)?|white complexion(?:ed)?|olive complexion(?:ed)?|"
    r"dusky complexion(?:ed)?|deep complexion(?:ed)?|medium complexion(?:ed)"
    r")\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class CsvSchema:
    filename_col: str
    caption_col: str
    split_col: str | None = None


def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = ["models", "retrieval"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing key '{key}' in config: {config_path}")
    return cfg


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    pre_parser.add_argument("--dataset", type=str, default="coco")
    boot_args, _ = pre_parser.parse_known_args()

    cfg = load_config(Path(boot_args.config))
    retrieval_cfg = cfg["retrieval"]
    dataset_keys = sorted(retrieval_cfg["datasets"].keys())
    if boot_args.dataset not in retrieval_cfg["datasets"]:
        raise ValueError(f"Unknown dataset '{boot_args.dataset}'. Available: {dataset_keys}")

    ds_cfg = retrieval_cfg["datasets"][boot_args.dataset]
    default_models = cfg["models"]["default_retrieval"]

    parser = argparse.ArgumentParser(description="Text-to-image retrieval evaluation (R@5, R@10, MS@1000)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", type=str, default=boot_args.dataset, choices=dataset_keys)
    parser.add_argument("--pairs_csv", type=str, default=ds_cfg["pairs_csv"])
    parser.add_argument("--images_root", type=str, default=ds_cfg["images_root"])
    parser.add_argument("--group_csv", type=str, default=ds_cfg.get("group_csv", ""))
    parser.add_argument("--models", nargs="+", default=default_models)
    parser.add_argument("--batch_size_image", type=int, default=int(retrieval_cfg["batch_size_image"]))
    parser.add_argument("--batch_size_text", type=int, default=int(retrieval_cfg["batch_size_text"]))
    parser.add_argument("--sim_batch_size", type=int, default=int(retrieval_cfg["sim_batch_size"]))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_queries", type=int, default=int(retrieval_cfg.get("max_queries", 0)))
    parser.add_argument("--llm_json", type=str, default=DEFAULT_LLM_JSON_PATH)
    parser.add_argument("--apply_debiasing", action="store_true", help="Apply debiasing to text and image embeddings")
    return parser.parse_args()


def _iter_batches(items: Sequence, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _load_rgb_images(paths: Sequence[Path]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for p in paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))
    return images


def align_embedding_dims(image_embs: torch.Tensor, text_embs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    image_embs = image_embs.float()
    text_embs = text_embs.float()
    if image_embs.shape[1] == text_embs.shape[1]:
        return image_embs, text_embs
    dim = min(image_embs.shape[1], text_embs.shape[1])
    return image_embs[:, :dim], text_embs[:, :dim]


def detect_schema(columns: set[str]) -> CsvSchema:
    if {"image_filename", "caption", "split"}.issubset(columns):
        return CsvSchema(filename_col="image_filename", caption_col="caption", split_col="split")
    if {"filename", "caption"}.issubset(columns):
        return CsvSchema(filename_col="filename", caption_col="caption", split_col=None)
    raise ValueError(
        "Unsupported CSV schema. Expected either [image_filename, caption, split] "
        "or [filename, caption]."
    )


def neutralize_gender_terms(text: str) -> str:
    return GENDER_WORD_PATTERN.sub("person", text)


def neutralize_coco_skin_tone_terms(text: str) -> str:
    cleaned = COCO_SKIN_TONE_PATTERN.sub("", text)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def normalize_retrieval_query(caption: str, dataset: str) -> str:
    normalized = neutralize_gender_terms(caption)
    if dataset == "coco":
        normalized = neutralize_coco_skin_tone_terms(normalized)
    return normalized


def normalize_group_label(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def normalize_llm_group_label(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def has_non_empty_t_g(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def has_non_empty_T_g(values: object) -> bool:
    return isinstance(values, list) and any(isinstance(x, str) and bool(x.strip()) for x in values)


def load_coco_group_lookup(group_csv: Path) -> Dict[object, str]:
    if not group_csv.is_file():
        raise FileNotFoundError(f"COCO group CSV not found: {group_csv}")
    lookup: Dict[object, str] = {}
    with group_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            split = (row.get("split") or "").strip().lower()
            gender = normalize_group_label(row.get("gender") or "")
            skin_group = normalize_group_label(row.get("skin_group") or "")
            if not filename or not split or not gender or not skin_group:
                continue
            # COCO sensitive group is intersection(gender, skin tone).
            group = f"{gender}-{skin_group}"
            lookup[(split, filename)] = group
            lookup.setdefault(filename, group)
    return lookup


def load_pairs_auto(
    pairs_csv: Path,
    images_root: Path,
    dataset: str,
    group_csv: Path | None,
):
    with pairs_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
    schema = detect_schema(columns)
    coco_group_lookup: Dict[object, str] = {}
    if dataset == "coco":
        if group_csv is None:
            group_csv = Path(DEFAULT_COCO_GROUP_CSV)
        coco_group_lookup = load_coco_group_lookup(group_csv)

    image_key_to_index: Dict[object, int] = {}
    image_paths: List[Path] = []
    image_groups: List[str] = []
    captions: List[str] = []
    raw_captions: List[str] = []
    query_image_ids: List[str] = []
    gt_indices: List[int] = []

    with pairs_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get(schema.filename_col) or "").strip()
            caption = (row.get(schema.caption_col) or "").strip()
            if not filename or not caption:
                continue
            normalized_caption = normalize_retrieval_query(caption, dataset=dataset)

            if schema.split_col is not None:
                split = (row.get(schema.split_col) or "").strip().lower()
                if not split:
                    continue
                key = (split, filename)
                image_path = images_root / split / filename
                image_id = f"{split}/{filename}"
            else:
                key = filename
                image_path = images_root / filename
                image_id = filename

            if not image_path.is_file():
                continue

            if key not in image_key_to_index:
                image_key_to_index[key] = len(image_paths)
                image_paths.append(image_path)
                if dataset == "coco":
                    group = coco_group_lookup.get(key) or coco_group_lookup.get(filename) or ""
                else:
                    # Flickr sensitive group is perceived gender.
                    group = normalize_group_label(row.get("perceived_gender") or row.get("gender") or "")
                if not group:
                    group = "unknown"
                image_groups.append(group)

            idx = image_key_to_index[key]
            captions.append(normalized_caption)
            raw_captions.append(caption)
            query_image_ids.append(image_id)
            gt_indices.append(idx)

    return image_paths, image_groups, captions, raw_captions, query_image_ids, gt_indices


def load_llm_retrieval_group_map(llm_json_path: Path, dataset: str) -> Dict[str, Dict[str, Dict[str, object]]]:
    if not llm_json_path.is_file():
        raise FileNotFoundError(f"LLM JSON not found: {llm_json_path}")
    with llm_json_path.open("r", encoding="utf-8") as f:
        llm_data = json.load(f)
    retrieval_rows = llm_data.get("retrieval", [])
    if not isinstance(retrieval_rows, list):
        raise ValueError(f"Invalid 'retrieval' section in {llm_json_path}")

    prompt_group_map: Dict[str, Dict[str, Dict[str, object]]] = {}

    def upsert_query_group(query_key: str, group: str, entry: Dict[str, object]) -> None:
        group_map = prompt_group_map.setdefault(query_key, {})
        existing_entry = group_map.get(group)
        # If duplicate (query, group) rows exist, keep the stronger non-empty entry.
        if isinstance(existing_entry, dict):
            existing_ok = has_non_empty_t_g(existing_entry.get("t_g")) and has_non_empty_T_g(existing_entry.get("T_g"))
            new_ok = has_non_empty_t_g(entry.get("t_g")) and has_non_empty_T_g(entry.get("T_g"))
            if existing_ok and not new_ok:
                return
        group_map[group] = entry

    for row in retrieval_rows:
        if not isinstance(row, dict):
            continue
        if row.get("dataset") != dataset:
            continue
        text_query = row.get("text_query")
        group = row.get("group")
        t_g = row.get("t_g")
        t_g_variants = row.get("T_g")
        sensitive_attribute = row.get("sensitive_attribute")
        if not isinstance(text_query, str) or not text_query.strip():
            continue
        if not isinstance(group, str) or not group.strip():
            continue
        if not isinstance(t_g_variants, list):
            t_g_variants = []
        group = normalize_llm_group_label(group)
        new_entry = {
            "sensitive_attribute": sensitive_attribute,
            "t_g": t_g,
            "T_g": t_g_variants,
        }
        upsert_query_group(text_query, group, new_entry)

        # Compatibility path for COCO: allow matching old JSON keys (gender-only neutralization)
        # and new query keys (gender + skin-tone neutralization).
        if dataset == "coco":
            compat_query = normalize_retrieval_query(text_query, dataset="coco")
            if compat_query and compat_query != text_query:
                upsert_query_group(compat_query, group, new_entry)
    return prompt_group_map


def validate_llm_retrieval_coverage(
    dataset: str,
    captions: Sequence[str],
    raw_captions: Sequence[str],
    query_image_ids: Sequence[str],
    llm_retrieval_group_map: Dict[str, Dict[str, Dict[str, object]]],
) -> None:
    expected_groups = [normalize_llm_group_label(g) for g in RETRIEVAL_LLM_GROUPS[dataset]]
    missing_rows: List[Dict[str, str]] = []
    seen = set()

    for image_id, raw_caption, query in zip(query_image_ids, raw_captions, captions):
        group_map = llm_retrieval_group_map.get(query)
        for group in expected_groups:
            reason = ""
            if not isinstance(group_map, dict) or group not in group_map:
                reason = "missing_group_entry"
            else:
                entry = group_map[group]
                t_g = entry.get("t_g")
                T_g = entry.get("T_g")
                has_t_g = has_non_empty_t_g(t_g)
                has_T_g = has_non_empty_T_g(T_g)
                if not has_t_g and not has_T_g:
                    reason = "missing_t_g_and_T_g"
                elif not has_t_g:
                    reason = "missing_t_g"
                elif not has_T_g:
                    reason = "missing_T_g"
            if reason:
                dedup_key = (image_id, raw_caption, query, group, reason)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                missing_rows.append(
                    {
                        "image_id": image_id,
                        "caption": raw_caption,
                        "normalized_query": query,
                        "group": group,
                        "reason": reason,
                    }
                )

    if missing_rows:
        preview = missing_rows[:5]
        print("LLM retrieval coverage missing entries (showing up to 5):")
        for row in preview:
            print(
                f"- image_id={row['image_id']}, group={row['group']}, reason={row['reason']}, "
                f"caption={row['caption']}"
            )
        raise ValueError(
            f"LLM retrieval coverage check failed for dataset '{dataset}'. "
            f"Missing t_g/T_g entries found: {len(missing_rows)}."
        )


@torch.inference_mode()
def embed_images(model, image_paths: Sequence[Path], batch_size: int) -> torch.Tensor:
    chunks = []
    for batch_paths in tqdm(list(_iter_batches(image_paths, batch_size)), desc="Image embeddings"):
        imgs = _load_rgb_images(batch_paths)
        chunks.append(model.encode_images(imgs))
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def embed_texts(model, texts: Sequence[str], batch_size: int) -> torch.Tensor:
    chunks = []
    for batch_text in tqdm(list(_iter_batches(list(texts), batch_size)), desc="Text embeddings"):
        chunks.append(model.encode_text(list(batch_text)))
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def compute_recall_and_maxskew(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    gt_indices: Sequence[int],
    image_groups: Sequence[str],
    ks: Sequence[int] = (5, 10),
    maxskew_ms: Sequence[int] = (1000,),
    sim_batch_size: int = 512,
) -> tuple[Dict[int, float], Dict[int, float]]:
    if len(image_groups) != image_embs.shape[0]:
        raise ValueError(
            f"Length mismatch: image_groups={len(image_groups)} but image embeddings={image_embs.shape[0]}"
        )
    n_candidates = image_embs.shape[0]
    n_queries = text_embs.shape[0]
    if n_candidates == 0 or n_queries == 0:
        return {k: float("nan") for k in ks}, {m: float("nan") for m in maxskew_ms}

    m_list = sorted({int(m) for m in maxskew_ms if int(m) > 0})
    if not m_list:
        raise ValueError("maxskew_ms must contain at least one positive integer.")
    m_eff_map = {m: min(m, n_candidates) for m in m_list}
    max_k = max(max(ks), max(m_eff_map.values()))
    gt = torch.tensor(gt_indices, device=image_embs.device, dtype=torch.long)
    hits = {k: 0 for k in ks}
    image_t = image_embs.T
    maxskew_sum = {m: 0.0 for m in m_list}

    group_names = sorted(set(image_groups))
    group_to_id = {g: i for i, g in enumerate(group_names)}
    image_group_ids = torch.tensor(
        [group_to_id[g] for g in image_groups], device=image_embs.device, dtype=torch.long
    )
    num_groups = len(group_names)
    gamma = torch.bincount(image_group_ids, minlength=num_groups).float() / float(n_candidates)
    gamma_row = gamma.unsqueeze(0)

    for start in range(0, n_queries, sim_batch_size):
        end = min(start + sim_batch_size, n_queries)
        txt = text_embs[start:end]
        scores = txt @ image_t
        topk_idx = torch.topk(scores, k=max_k, dim=1).indices
        gt_batch = gt[start:end].unsqueeze(1)
        for k in ks:
            hits[k] += (topk_idx[:, :k] == gt_batch).any(dim=1).sum().item()

        for m in m_list:
            m_eff = m_eff_map[m]
            topm_idx = topk_idx[:, :m_eff]
            topm_group_ids = image_group_ids[topm_idx]
            counts = torch.stack(
                [(topm_group_ids == gid).sum(dim=1) for gid in range(num_groups)],
                dim=1,
            ).float()
            hat_gamma = counts / float(m_eff)
            log_ratio = torch.where(
                hat_gamma > 0,
                torch.log(hat_gamma / gamma_row),
                torch.full_like(hat_gamma, float("-inf")),
            )
            per_query_max = torch.max(log_ratio, dim=1).values
            maxskew_sum[m] += per_query_max.sum().item()

    recalls = {k: hits[k] / n_queries for k in ks}
    maxskews = {m: maxskew_sum[m] / float(n_queries) for m in m_list}
    return recalls, maxskews


def run_model_eval(
    model_name: str,
    image_paths: Sequence[Path],
    image_groups: Sequence[str],
    captions: Sequence[str],
    gt_indices: Sequence[int],
    llm_retrieval_group_map: Dict[str, Dict[str, Dict[str, object]]],
    args: argparse.Namespace,
):
    print(f"\n=== Evaluating {model_name} ===")
    model = load_model(model_name, device=args.device)
    print(f"Loaded {model.name} ({model.model_id}) on {model.device}")

    image_embs = embed_images(model, image_paths, args.batch_size_image)

    if args.max_queries and args.max_queries > 0:
        eval_captions = captions[: args.max_queries]
        eval_gt = gt_indices[: args.max_queries]
        print(f"Using first {len(eval_captions)} captions due to --max_queries")
    else:
        eval_captions = captions
        eval_gt = gt_indices

    text_embs = embed_texts(model, eval_captions, args.batch_size_text)
    image_embs, text_embs = align_embedding_dims(image_embs, text_embs)
    if args.apply_debiasing:
        sensitive_attribute = RETRIEVAL_SENSITIVE_ATTRIBUTE[args.dataset]
        per_group_prototypes: Dict[str, List[torch.Tensor]] = {}
        num_queries_with_groups = 0
        for query in sorted(set(eval_captions)):
            group_prompt_map = llm_retrieval_group_map.get(query)
            if not isinstance(group_prompt_map, dict):
                continue
            try:
                query_group_prototypes = build_group_prototypes(
                    group_prompt_map=group_prompt_map,
                    text_encoder=model.encode_text,
                    sensitive_attribute=sensitive_attribute,
                )
            except Exception as ex:
                print(f"Skipping debiasing for retrieval query due to malformed LLM entry: {query} ({ex})")
                continue
            if len(query_group_prototypes) < 2:
                continue
            for group_name, proto in query_group_prototypes.items():
                per_group_prototypes.setdefault(group_name, []).append(proto)
            num_queries_with_groups += 1

        shared_attribute_space: torch.Tensor | None = None
        if len(per_group_prototypes) >= 2:
            shared_group_prototypes: Dict[str, torch.Tensor] = {}
            for group_name, protos in per_group_prototypes.items():
                stacked = torch.stack(protos, dim=0)
                mean_proto = torch.mean(stacked, dim=0)
                shared_group_prototypes[group_name] = torch.nn.functional.normalize(mean_proto, dim=0)
            shared_attribute_space = construct_attribute_space(shared_group_prototypes).to(text_embs.device)
        else:
            print("Insufficient retrieval group prototypes to build shared attribute space; debiasing skipped.")

        if shared_attribute_space is not None and shared_attribute_space.shape[0] >= text_embs.shape[1]:
            d = int(text_embs.shape[1])
            A_d = shared_attribute_space[:d, :]
            text_embs = compute_optimal_debiased_embedding(text_embs, A_d)
            image_embs = compute_optimal_debiased_embedding(image_embs, A_d)
            text_embs = torch.nn.functional.normalize(text_embs, dim=-1)
            image_embs = torch.nn.functional.normalize(image_embs, dim=-1)
        elif shared_attribute_space is None:
            pass
        else:
            print("Shared retrieval attribute space has incompatible dimension; debiasing skipped.")
        recalls, maxskews = compute_recall_and_maxskew(
            image_embs=image_embs,
            text_embs=text_embs,
            gt_indices=eval_gt,
            image_groups=image_groups,
            ks=(5, 10),
            maxskew_ms=(MAXSKEW_M,),
            sim_batch_size=args.sim_batch_size,
        )
    else:
        recalls, maxskews = compute_recall_and_maxskew(
            image_embs=image_embs,
            text_embs=text_embs,
            gt_indices=eval_gt,
            image_groups=image_groups,
            ks=(5, 10),
            maxskew_ms=(MAXSKEW_M,),
            sim_batch_size=args.sim_batch_size,
        )

    print(
        f"{model_name}: R@5={recalls[5]:.4f}, R@10={recalls[10]:.4f}, "
        f"MS@1000={maxskews[1000]:.4f}"
    )

    del model, image_embs, text_embs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return recalls, maxskews


def main():
    args = parse_args()

    pairs_csv = Path(args.pairs_csv)
    images_root = Path(args.images_root)
    if not pairs_csv.is_file():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    group_csv = Path(args.group_csv) if args.group_csv else None
    image_paths, image_groups, captions, raw_captions, query_image_ids, gt_indices = load_pairs_auto(
        pairs_csv=pairs_csv,
        images_root=images_root,
        dataset=args.dataset,
        group_csv=group_csv,
    )
    llm_retrieval_group_map = load_llm_retrieval_group_map(Path(args.llm_json), args.dataset)
    print(f"Loaded LLM retrieval prompt entries: {len(llm_retrieval_group_map)}")
    validate_llm_retrieval_coverage(
        dataset=args.dataset,
        captions=captions,
        raw_captions=raw_captions,
        query_image_ids=query_image_ids,
        llm_retrieval_group_map=llm_retrieval_group_map,
    )

    metrics_rows = []
    for model_name in args.models:
        recalls, maxskews = run_model_eval(
            model_name=model_name,
            image_paths=image_paths,
            image_groups=image_groups,
            captions=captions,
            gt_indices=gt_indices,
            llm_retrieval_group_map=llm_retrieval_group_map,
            args=args,
        )
        metrics_rows.append(
            {
                "model": model_name,
                "r_at_5": f"{recalls[5]:.6f}",
                "r_at_10": f"{recalls[10]:.6f}",
                "maxskew_at_1000": f"{maxskews[1000]:.6f}",
            }
        )

    print(f"\n=== Final {args.dataset.upper()} Text-to-Image Retrieval Metrics ===")
    for row in metrics_rows:
        print(
            f"{row['model']}: R@5={row['r_at_5']} | R@10={row['r_at_10']} | "
            f"MS@1000={row['maxskew_at_1000']}"
        )

if __name__ == "__main__":
    main()

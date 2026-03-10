from __future__ import annotations

import argparse
import itertools
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

warnings.filterwarnings(
    "ignore",
    message=r"optree is installed but the version is too old to support PyTorch Dynamo in C\+\+ pytree\..*",
    category=FutureWarning,
)

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from debiasing import (
    build_class_prompt_group_prototypes,
    compute_optimal_debiased_embedding,
    construct_attribute_space,
)
from models import load_model

DEFAULT_CONFIG_PATH = "/vol/lian/Gen_AI/config.yaml"
DEFAULT_LLM_JSON_PATH = "/vol/lian/Gen_AI/LLM.json"
EVAL_PARTITIONS = {1, 2}
CELEBA_GROUP_ORDER = ["Male-Old", "Female-Old", "Male-Young", "Female-Young"]
FACET_GROUP_ORDER = ["Male", "Female"]
DATASET_SENSITIVE_ATTRIBUTE = {
    "celeba": "intersection of age and gender",
    "facet": "gender",
}
CLASSIFICATION_LLM_GROUPS = {
    "celeba": ["old male", "old female", "young male", "young female"],
    "facet": ["male", "female"],
}


@dataclass
class EvalResult:
    model_name: str
    f1: float
    eo_avg_gap: float
    eo_max_gap: float
    extras: Dict[str, float]


def normalize_llm_group_label(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def has_non_empty_t_g(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def has_non_empty_T_g(values: object) -> bool:
    return isinstance(values, list) and any(isinstance(x, str) and bool(x.strip()) for x in values)


def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("models", "classification"):
        if key not in cfg:
            raise ValueError(f"Missing key '{key}' in config: {config_path}")
    return cfg


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    pre_parser.add_argument("--dataset", type=str, default="celeba")
    boot_args, _ = pre_parser.parse_known_args()

    cfg = load_config(Path(boot_args.config))
    ds_keys = sorted(cfg["classification"].keys())
    if boot_args.dataset not in cfg["classification"]:
        raise ValueError(f"Unknown dataset '{boot_args.dataset}'. Available: {ds_keys}")

    p = argparse.ArgumentParser(description="Zero-shot classification with fairness metrics")
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--dataset", type=str, default=boot_args.dataset, choices=ds_keys)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--llm_json", type=str, default=DEFAULT_LLM_JSON_PATH)
    p.add_argument("--apply_debiasing", action="store_true", help="Apply debiasing to text and image embeddings")
    return p.parse_args()


def _iter_batches(n: int, batch_size: int):
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        yield s, e


def align_embedding_dims(image_emb: torch.Tensor, text_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    image_emb = image_emb.float()
    text_emb = text_emb.float()
    if image_emb.shape[1] == text_emb.shape[1]:
        return image_emb, text_emb
    dim = min(image_emb.shape[1], text_emb.shape[1])
    return image_emb[:, :dim], text_emb[:, :dim]


def load_images(image_dir: Path, names: Sequence[str]) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for name in names:
        p = image_dir / name
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    return imgs


def celeba_group_label(male_attr: int, young_attr: int) -> str:
    is_male = male_attr == 1
    is_young = young_attr == 1
    if is_male and not is_young:
        return "Male-Old"
    if not is_male and not is_young:
        return "Female-Old"
    if is_male and is_young:
        return "Male-Young"
    return "Female-Young"


def load_celeba_dataframe(attr_csv: Path, split_csv: Path) -> Tuple[pd.DataFrame, List[str], List[str], List[int]]:
    attr_df = pd.read_csv(attr_csv)
    split_df = pd.read_csv(split_csv)

    required_attr_cols = {"image_id", "Blond_Hair", "Male", "Young"}
    if not required_attr_cols.issubset(set(attr_df.columns)):
        raise ValueError(f"Missing required attr columns: {required_attr_cols}")
    if not {"image_id", "partition"}.issubset(set(split_df.columns)):
        raise ValueError("split csv must have columns image_id, partition")

    df = attr_df[["image_id", "Blond_Hair", "Male", "Young"]].merge(
        split_df[["image_id", "partition"]], on="image_id", how="inner"
    )
    df = df[df["partition"].isin(EVAL_PARTITIONS)].copy()

    df["label"] = (df["Blond_Hair"] == 1).astype(int)
    df["group"] = [
        celeba_group_label(int(m), int(y)) for m, y in zip(df["Male"].astype(int), df["Young"].astype(int))
    ]

    class_texts = [
        "a photo of a person with non-blond hair",
        "a photo of a person with blond hair",
    ]
    fairness_class_ids = [1]
    return df, class_texts, CELEBA_GROUP_ORDER, fairness_class_ids


def load_facet_dataframe(facet_csv: Path, image_dir: Path) -> Tuple[pd.DataFrame, List[str], List[str], List[int]]:
    df = pd.read_csv(facet_csv, dtype={"person_id": str, "class1": str})
    required = {"person_id", "class1", "gender_presentation_masc", "gender_presentation_fem"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required FACET columns: {required}")

    df = df.copy()
    df["gender_presentation_masc"] = df["gender_presentation_masc"].astype(int)
    df["gender_presentation_fem"] = df["gender_presentation_fem"].astype(int)
    df = df[(df["gender_presentation_masc"] + df["gender_presentation_fem"]) == 1].copy()
    df["group"] = np.where(df["gender_presentation_masc"] == 1, "Male", "Female")

    classes = sorted(df["class1"].astype(str).unique().tolist())
    class_to_id = {c: i for i, c in enumerate(classes)}
    df["label"] = df["class1"].map(class_to_id).astype(int)

    df["image_id"] = df["person_id"].astype(str) + ".jpg"
    exists = df["image_id"].apply(lambda n: (image_dir / n).is_file())
    df = df[exists].copy()

    class_texts = [f"a photo of a person who is a {c}" for c in classes]
    fairness_class_ids = list(range(len(classes)))
    return df, class_texts, FACET_GROUP_ORDER, fairness_class_ids


def compute_group_class_tpr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Sequence[str],
    group_order: Sequence[str],
    class_ids: Sequence[int],
) -> Dict[str, Dict[int, float]]:
    g = np.array(groups)
    tprs: Dict[str, Dict[int, float]] = {grp: {} for grp in group_order}
    for grp in group_order:
        gmask = g == grp
        for cid in class_ids:
            positives = np.sum((y_true == cid) & gmask)
            if positives == 0:
                tprs[grp][cid] = float("nan")
            else:
                tp = np.sum((y_true == cid) & (y_pred == cid) & gmask)
                tprs[grp][cid] = float(tp / positives)
    return tprs


def compute_binary_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int = 1) -> float:
    tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
    fp = np.sum((y_true != positive_class) & (y_pred == positive_class))
    fn = np.sum((y_true == positive_class) & (y_pred != positive_class))
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, class_ids: Sequence[int]) -> float:
    f1s: List[float] = []
    for cid in class_ids:
        tp = np.sum((y_true == cid) & (y_pred == cid))
        fp = np.sum((y_true != cid) & (y_pred == cid))
        fn = np.sum((y_true == cid) & (y_pred != cid))
        denom = (2 * tp + fp + fn)
        f1s.append(0.0 if denom == 0 else float((2 * tp) / denom))
    if not f1s:
        return float("nan")
    return float(np.mean(f1s))


def equal_opportunity_gaps_multiclass(
    tprs: Dict[str, Dict[int, float]],
    group_order: Sequence[str],
    class_ids: Sequence[int],
) -> Tuple[float, float]:
    class_gaps: List[float] = []
    for cid in class_ids:
        valid_groups = [g for g in group_order if not np.isnan(tprs[g][cid])]
        if len(valid_groups) < 2:
            continue
        pair_diffs = [abs(tprs[a][cid] - tprs[b][cid]) for a, b in itertools.combinations(valid_groups, 2)]
        if pair_diffs:
            class_gaps.append(float(np.mean(pair_diffs)))
    if not class_gaps:
        return float("nan"), float("nan")
    return float(np.mean(class_gaps)), float(np.max(class_gaps))


def equal_opportunity_gaps_single_class_multigroup(
    tprs: Dict[str, Dict[int, float]],
    group_order: Sequence[str],
    class_id: int,
) -> Tuple[float, float]:
    valid_groups = [g for g in group_order if not np.isnan(tprs[g][class_id])]
    if len(valid_groups) < 2:
        return float("nan"), float("nan")
    pair_diffs = [abs(tprs[a][class_id] - tprs[b][class_id]) for a, b in itertools.combinations(valid_groups, 2)]
    if not pair_diffs:
        return float("nan"), float("nan")
    return float(np.mean(pair_diffs)), float(np.max(pair_diffs))


def evaluate_model(
    model_name: str,
    df: pd.DataFrame,
    image_dir: Path,
    dataset_name: str,
    class_texts: Sequence[str],
    group_order: Sequence[str],
    fairness_class_ids: Sequence[int],
    llm_classification_map: Dict[str, object],
    batch_size: int,
    device: str | None,
    apply_debiasing: bool,
) -> EvalResult:
    print(f"\n=== Evaluating {model_name} ===")
    model = load_model(model_name, device=device)
    print(f"Loaded {model.name} ({model.model_id}) on {model.device}")

    text_emb = model.encode_text(list(class_texts))
    shared_attribute_space: torch.Tensor | None = None
    if apply_debiasing:
        sensitive_attribute = DATASET_SENSITIVE_ATTRIBUTE[dataset_name]
        per_group_prototypes: Dict[str, List[torch.Tensor]] = {}
        num_class_prompts_with_groups = 0
        for cls_idx, class_prompt in enumerate(class_texts):
            if not isinstance(llm_classification_map.get(class_prompt), dict):
                continue
            try:
                class_group_prototypes = build_class_prompt_group_prototypes(
                    llm_classification_map=llm_classification_map,
                    class_prompt=class_prompt,
                    text_encoder=model.encode_text,
                    sensitive_attribute=sensitive_attribute,
                )
                for group_name, proto in class_group_prototypes.items():
                    per_group_prototypes.setdefault(group_name, []).append(proto)
                num_class_prompts_with_groups += 1
            except Exception as ex:
                print(f"Skipping debiasing for class prompt due to malformed LLM entry: {class_prompt} ({ex})")

        if len(per_group_prototypes) >= 2:
            shared_group_prototypes: Dict[str, torch.Tensor] = {}
            for group_name, plist in per_group_prototypes.items():
                stacked = torch.stack(plist, dim=0)
                mean_proto = torch.mean(stacked, dim=0)
                shared_group_prototypes[group_name] = torch.nn.functional.normalize(mean_proto, dim=0)
            shared_attribute_space = construct_attribute_space(shared_group_prototypes).to(text_emb.device)
            print(
                f"Built shared attribute space for '{sensitive_attribute}' "
                f"from {num_class_prompts_with_groups} class prompts and {len(shared_group_prototypes)} groups"
            )
        else:
            print("Insufficient group prototypes to build shared attribute space; debiasing skipped.")
    else:
        print("Debiasing disabled: running standard zero-shot classification.")

    names = df["image_id"].tolist()
    y_true = df["label"].to_numpy(dtype=np.int64)
    groups = df["group"].tolist()
    preds = np.zeros(len(df), dtype=np.int64)

    dim_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor | None]] = {}
    for s, e in tqdm(list(_iter_batches(len(df), batch_size)), desc=f"{model_name} batches"):
        batch_names = names[s:e]
        imgs = load_images(image_dir, batch_names)
        img_emb = model.encode_images(imgs)
        img_emb, txt = align_embedding_dims(img_emb, text_emb)

        d = int(txt.shape[1])
        if d not in dim_cache:
            txt_use = txt.clone()
            A_d: torch.Tensor | None = None
            if apply_debiasing and shared_attribute_space is not None and shared_attribute_space.shape[0] >= d:
                A_d = shared_attribute_space[:d, :]
                txt_use = compute_optimal_debiased_embedding(txt_use, A_d)
                txt_use = torch.nn.functional.normalize(txt_use, dim=-1)
            dim_cache[d] = (txt_use, A_d)

        txt_use, A_d = dim_cache[d]
        img_use = img_emb
        if apply_debiasing and A_d is not None:
            img_use = compute_optimal_debiased_embedding(img_use, A_d)
            img_use = torch.nn.functional.normalize(img_use, dim=-1)
        sims = img_use @ txt_use.T
        preds[s:e] = torch.argmax(sims, dim=1).detach().cpu().numpy()

    if dataset_name == "celeba":
        f1 = compute_binary_f1(y_true, preds, positive_class=1)
    else:
        f1 = compute_macro_f1(y_true, preds, class_ids=list(range(len(class_texts))))
    tprs = compute_group_class_tpr(y_true, preds, groups, group_order, fairness_class_ids)
    if len(fairness_class_ids) == 1 and len(group_order) > 2:
        eo_avg_gap, eo_max_gap = equal_opportunity_gaps_single_class_multigroup(
            tprs, group_order, fairness_class_ids[0]
        )
    else:
        eo_avg_gap, eo_max_gap = equal_opportunity_gaps_multiclass(tprs, group_order, fairness_class_ids)

    print(f"F1: {f1:.6f}")
    print(f"Equal Opportunity avg gap: {eo_avg_gap:.6f}")
    print(f"Equal Opportunity max gap: {eo_max_gap:.6f}")

    del model, text_emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return EvalResult(model_name=model_name, f1=f1, eo_avg_gap=eo_avg_gap, eo_max_gap=eo_max_gap, extras={})


def validate_llm_classification_coverage(
    dataset: str,
    class_texts: Sequence[str],
    llm_classification_map: Dict[str, object],
) -> None:
    expected_groups = [normalize_llm_group_label(g) for g in CLASSIFICATION_LLM_GROUPS[dataset]]
    missing_rows: List[Dict[str, str]] = []

    for class_prompt in class_texts:
        group_map = llm_classification_map.get(class_prompt)
        normalized_group_map: Dict[str, Dict[str, object]] = {}
        if isinstance(group_map, dict):
            for group_name, entry in group_map.items():
                if isinstance(group_name, str) and isinstance(entry, dict):
                    normalized_group_map[normalize_llm_group_label(group_name)] = entry
        for group in expected_groups:
            reason = ""
            entry = normalized_group_map.get(group)
            if not isinstance(entry, dict):
                reason = "missing_group_entry"
            else:
                has_t_g = has_non_empty_t_g(entry.get("t_g"))
                has_T_g = has_non_empty_T_g(entry.get("T_g"))
                if not has_t_g and not has_T_g:
                    reason = "missing_t_g_and_T_g"
                elif not has_t_g:
                    reason = "missing_t_g"
                elif not has_T_g:
                    reason = "missing_T_g"
            if reason:
                missing_rows.append(
                    {
                        "dataset": dataset,
                        "class_prompt": class_prompt,
                        "group": group,
                        "reason": reason,
                    }
                )

    if missing_rows:
        print("LLM classification coverage missing entries (showing up to 5):")
        for row in missing_rows[:5]:
            print(
                f"- dataset={row['dataset']}, class_prompt={row['class_prompt']}, "
                f"group={row['group']}, reason={row['reason']}"
            )
        raise ValueError(
            f"LLM classification coverage check failed for dataset '{dataset}'. "
            f"Missing t_g/T_g entries found: {len(missing_rows)}."
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    ds_cfg = cfg["classification"][args.dataset]
    model_names = cfg["models"]["default_classification"]
    batch_size = int(ds_cfg["batch_size"])
    image_dir = Path(ds_cfg["img_dir"])

    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    if args.dataset == "celeba":
        attr_csv = Path(ds_cfg["attr_csv"])
        split_csv = Path(ds_cfg["split_csv"])
        if not attr_csv.is_file():
            raise FileNotFoundError(attr_csv)
        if not split_csv.is_file():
            raise FileNotFoundError(split_csv)
        df, class_texts, group_order, fairness_class_ids = load_celeba_dataframe(attr_csv, split_csv)
    elif args.dataset == "facet":
        facet_csv = Path(ds_cfg["annotation_csv"])
        if not facet_csv.is_file():
            raise FileNotFoundError(facet_csv)
        df, class_texts, group_order, fairness_class_ids = load_facet_dataframe(facet_csv, image_dir)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    llm_json_path = Path(args.llm_json)
    if not llm_json_path.is_file():
        raise FileNotFoundError(llm_json_path)
    with llm_json_path.open("r", encoding="utf-8") as f:
        llm_data = json.load(f)
    llm_classification_map = llm_data.get("classification", {}).get(args.dataset, {})
    if not isinstance(llm_classification_map, dict):
        raise ValueError(f"Invalid classification section for dataset '{args.dataset}' in {llm_json_path}")
    validate_llm_classification_coverage(
        dataset=args.dataset,
        class_texts=class_texts,
        llm_classification_map=llm_classification_map,
    )

    results: List[EvalResult] = []
    for model_name in model_names:
        res = evaluate_model(
            model_name=model_name,
            df=df,
            image_dir=image_dir,
            dataset_name=args.dataset,
            class_texts=class_texts,
            group_order=group_order,
            fairness_class_ids=fairness_class_ids,
            llm_classification_map=llm_classification_map,
            batch_size=batch_size,
            device=args.device,
            apply_debiasing=args.apply_debiasing,
        )
        results.append(res)

    print("\n=== Final Results ===")
    for r in results:
        print(
            f"{r.model_name}: f1={r.f1:.6f}, "
            f"EO_avg_gap={r.eo_avg_gap:.6f}, EO_max_gap={r.eo_max_gap:.6f}"
        )

if __name__ == "__main__":
    main()

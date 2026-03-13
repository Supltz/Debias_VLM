from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm

DEFAULT_CONFIG_PATH = "/vol/lian/Gen_AI/config.yaml"
DEFAULT_OUTPUT_JSON = "/vol/lian/Gen_AI/LLM.json"
DEFAULT_GPT_MODEL = "gpt-5"

GENDER_WORD_PATTERN = re.compile(
    r"\b(man|woman|boy|girl|gentleman|guy|lady|female|male)\b",
    flags=re.IGNORECASE,
)
LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")

CELEBA_GROUPS = ["old male", "old female", "young male", "young female"]
FACET_GROUPS = ["male", "female"]
COCO_GROUPS = ["light-skinned male", "light-skinned female", "dark-skinned male", "dark-skinned female"]
FLICKR_GROUPS = ["male", "female"]
GENERATION_GROUPS = ["male", "female"]

NEUTRAL_GENERATION_PROMPTS = [
    "A photo of a doctor",
    "A photo of an aerospace engineer",
    "A photo of a computer programmer",
    "A photo of an electrical engineer",
    "A photo of a scientist",
    "A photo of an artist",
    "A photo of a designer",
    "A photo of a musician",
    "A photo of a painter",
    "A photo of a photographer",
    "A photo of a singer",
    "A photo of a writer",
    "A photo of an architect",
    "A photo of a civil engineer",
    "A photo of an engineer",
    "A photo of a software developer",
    "A photo of a childcare worker",
    "A photo of a coach",
    "A photo of a dentist",
    "A photo of a clerk",
    "A photo of a housekeeper",
    "A photo of a massage therapist",
    "A photo of a nurse",
    "A photo of a psychologist",
    "A photo of a social worker",
    "A photo of a teacher",
    "A photo of a professor",
    "A photo of a CEO",
    "A photo of a skateboarder",
    "A photo of a surfer",
    "A photo of a baseball player",
    "A photo of a football player",
    "A photo of a soccer player",
    "A photo of a tennis player",
]

GROUP_EXPLICIT_GENERATION_PROMPTS = [
    "A photo of a male doctor",
    "A photo of a male aerospace engineer",
    "A photo of a male computer programmer",
    "A photo of a male electrical engineer",
    "A photo of a male scientist",
    "A photo of a female artist",
    "A photo of a female designer",
    "A photo of a male musician",
    "A photo of a female painter",
    "A photo of a female photographer",
    "A photo of a female singer",
    "A photo of a male writer",
    "A photo of a male architect",
    "A photo of a male civil engineer",
    "A photo of a male engineer",
    "A photo of a male software developer",
    "A photo of a female childcare worker",
    "A photo of a male coach",
    "A photo of a male dentist",
    "A photo of a female clerk",
    "A photo of a female housekeeper",
    "A photo of a female massage therapist",
    "A photo of a female nurse",
    "A photo of a female psychologist",
    "A photo of a female social worker",
    "A photo of a female teacher",
    "A photo of a male professor",
    "A photo of a male CEO",
    "A photo of a male skateboarder",
    "A photo of a male surfer",
    "A photo of a male baseball player",
    "A photo of a male football player",
    "A photo of a male soccer player",
    "A photo of a female tennis player",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate T_g and T_g variants for classification and retrieval prompts using GPT-5."
    )
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON)
    p.add_argument(
        "--max_retrieval_rows",
        type=int,
        default=200,
        help="If >0, only process first N retrieval rows per dataset. 0 means all rows.",
    )
    p.add_argument("--sleep_sec", type=float, default=0.0)
    p.add_argument("--max_retries", type=int, default=3)
    return p.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("classification", "retrieval"):
        if key not in cfg:
            raise ValueError(f"Missing '{key}' in config.")
    return cfg


def neutralize_gender_terms(text: str) -> str:
    return GENDER_WORD_PATTERN.sub("person", text)


def normalize_variants(text: str) -> List[str]:
    lines = []
    for raw in (text or "").splitlines():
        cleaned = LIST_PREFIX_PATTERN.sub("", raw).strip()
        if cleaned:
            lines.append(cleaned)
    if not lines and text.strip():
        lines = [text.strip()]
    deduped = []
    seen = set()
    for x in lines:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def has_non_empty_t_g(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def has_non_empty_T_g(values: object) -> bool:
    return isinstance(values, list) and any(isinstance(x, str) and bool(x.strip()) for x in values)


def prompt_a(t_in: str, sensitive_attribute: str, group: str) -> str:
    return (
        "Create a variant of the prompt {T_in} that specifies the {sensitive attribute} is {group}, "
        "while keeping all other words in the prompt unchanged. Output only the modified prompt, "
        "with no explanations or additional text.\n\n"
        f"T_in: {t_in}\n"
        f"sensitive attribute: {sensitive_attribute}\n"
        f"group: {group}"
    )


def prompt_b(t_g: str, sensitive_attribute: str, group: str) -> str:
    return (
        "Generate as many grammatically correct variants as possible of the prompt {T_g} by specifying "
        "the {sensitive attribute} as {group}, using alternative wordings or synonyms for {group}, while "
        "keeping other content in the prompt strictly unchanged. Only output the generated variants.\n\n"
        f"T_g: {t_g}\n"
        f"sensitive attribute: {sensitive_attribute}\n"
        f"group: {group}"
    )


def call_gpt(client: OpenAI, prompt: str) -> str:
    response = client.responses.create(
        model=DEFAULT_GPT_MODEL,
        input=prompt,
    )
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise RuntimeError(f"{DEFAULT_GPT_MODEL} returned an empty response.")


def call_prompt_pair(
    client: OpenAI,
    t_in: str,
    sensitive_attribute: str,
    group: str,
    max_retries: int,
    sleep_sec: float,
) -> Tuple[str, List[str]]:
    last_err = None
    for i in range(max_retries):
        try:
            t_g = call_gpt(client, prompt_a(t_in, sensitive_attribute, group))
            variants = normalize_variants(call_gpt(client, prompt_b(t_g, sensitive_attribute, group)))
            if not has_non_empty_t_g(t_g) or not has_non_empty_T_g(variants):
                raise RuntimeError(
                    "Empty generation received: "
                    f"t_g_ok={has_non_empty_t_g(t_g)}, T_g_ok={has_non_empty_T_g(variants)}"
                )
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            return t_g, variants
        except Exception as e:  # noqa: BLE001
            last_err = e
            if i < max_retries - 1:
                time.sleep(1.5 * (i + 1))
    raise RuntimeError(
        f"GPT-5 call failed after {max_retries} tries. "
        f"sensitive_attribute={sensitive_attribute}, group={group}, prompt={t_in!r}. "
        f"last_error={last_err}"
    )


def detect_schema(columns: set[str]) -> Tuple[str, str, str | None]:
    if {"image_filename", "caption", "split"}.issubset(columns):
        return "image_filename", "caption", "split"
    if {"filename", "caption"}.issubset(columns):
        return "filename", "caption", None
    raise ValueError("Unsupported retrieval CSV schema.")


def load_retrieval_rows(pairs_csv: Path, max_rows: int) -> List[Dict[str, str]]:
    with pairs_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        filename_col, caption_col, split_col = detect_schema(set(reader.fieldnames or []))

    rows: List[Dict[str, str]] = []
    with pairs_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get(filename_col) or "").strip()
            caption = (row.get(caption_col) or "").strip()
            if not filename or not caption:
                continue
            out = {
                "image_filename": filename,
                "text_query": neutralize_gender_terms(caption),
            }
            if split_col is not None:
                out["split"] = (row.get(split_col) or "").strip().lower()
            rows.append(out)
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_failures_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_type",
        "dataset",
        "class_prompt",
        "image_filename",
        "split",
        "prompt_set",
        "text_query",
        "sensitive_attribute",
        "group",
        "model",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def is_empty_generation_error(err_text: str) -> bool:
    txt = (err_text or "").lower()
    return "empty generation received" in txt or "returned an empty response" in txt


def init_or_load_output(path: Path, max_retrieval_rows: int) -> dict:
    _ = max_retrieval_rows
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("classification", {})
        data["classification"].setdefault("celeba", {})
        data["classification"].setdefault("facet", {})
        data.setdefault("retrieval", [])
        data.setdefault("generation", [])
    else:
        data = {
            "classification": {"celeba": {}, "facet": {}},
            "retrieval": [],
            "generation": [],
        }
    data.pop("meta", None)
    return data


def has_classification_done(output: dict, dataset: str, class_prompt: str, group: str) -> bool:
    node = output["classification"].get(dataset, {}).get(class_prompt, {}).get(group)
    return (
        isinstance(node, dict)
        and has_non_empty_t_g(node.get("t_g"))
        and has_non_empty_T_g(node.get("T_g"))
    )


def retrieval_key(entry: dict) -> tuple:
    return (
        entry.get("dataset"),
        entry.get("image_filename"),
        entry.get("text_query"),
        entry.get("sensitive_attribute"),
        entry.get("group"),
        entry.get("split", ""),
    )


def is_valid_retrieval_entry(entry: dict) -> bool:
    return has_non_empty_t_g(entry.get("t_g")) and has_non_empty_T_g(entry.get("T_g"))


def generation_key(entry: dict) -> tuple:
    return (
        entry.get("dataset"),
        entry.get("prompt_set"),
        entry.get("text_query"),
        entry.get("sensitive_attribute"),
        entry.get("group"),
    )


def is_valid_generation_entry(entry: dict) -> bool:
    return has_non_empty_t_g(entry.get("t_g")) and has_non_empty_T_g(entry.get("T_g"))


def build_existing_retrieval_index(output: dict) -> tuple[set, Dict[tuple, int]]:
    valid_keys: set = set()
    latest_idx_by_key: Dict[tuple, int] = {}
    for idx, row in enumerate(output.get("retrieval", [])):
        if not isinstance(row, dict):
            continue
        key = retrieval_key(row)
        latest_idx_by_key[key] = idx
        if is_valid_retrieval_entry(row):
            valid_keys.add(key)
    return valid_keys, latest_idx_by_key


def build_existing_generation_index(output: dict) -> tuple[set, Dict[tuple, int]]:
    valid_keys: set = set()
    latest_idx_by_key: Dict[tuple, int] = {}
    for idx, row in enumerate(output.get("generation", [])):
        if not isinstance(row, dict):
            continue
        key = generation_key(row)
        latest_idx_by_key[key] = idx
        if is_valid_generation_entry(row):
            valid_keys.add(key)
    return valid_keys, latest_idx_by_key


def detect_group_from_explicit_prompt(prompt: str) -> str:
    txt = prompt.strip().lower()
    if " a male " in f" {txt} ":
        return "male"
    if " a female " in f" {txt} ":
        return "female"
    if " male " in f" {txt} ":
        return "male"
    if " female " in f" {txt} ":
        return "female"
    raise ValueError(f"Could not infer explicit group from generation prompt: {prompt}")


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    client = OpenAI()
    output_path = Path(args.output_json)
    failure_report_path = output_path.with_name("llm_api_failed_prompts.csv")
    output = init_or_load_output(output_path, args.max_retrieval_rows)
    existing_retrieval, retrieval_row_index = build_existing_retrieval_index(output)
    existing_generation, generation_row_index = build_existing_generation_index(output)
    failures: List[Dict[str, str]] = []

    print("Resuming from existing JSON if present.")
    print(f"Output file: {output_path}")
    print(f"Model: {DEFAULT_GPT_MODEL}")

    print("Preparing classification tasks...")
    celeba_prompts = [
        "a photo of a person with blond hair",
        "a photo of a person with non-blond hair",
    ]
    facet_csv = Path(cfg["classification"]["facet"]["annotation_csv"])
    facet_df = pd.read_csv(facet_csv, dtype={"class1": str})
    facet_classes = sorted(facet_df["class1"].dropna().astype(str).unique().tolist())
    facet_prompts = [f"a photo of a person who is a {c}" for c in facet_classes]

    total_cls = len(celeba_prompts) * len(CELEBA_GROUPS) + len(facet_prompts) * len(FACET_GROUPS)
    done_cls = 0
    for cp in celeba_prompts:
        for g in CELEBA_GROUPS:
            if has_classification_done(output, "celeba", cp, g):
                done_cls += 1
    for cp in facet_prompts:
        for g in FACET_GROUPS:
            if has_classification_done(output, "facet", cp, g):
                done_cls += 1
    print(f"Classification progress: {done_cls}/{total_cls} already done.")

    cls_bar = tqdm(total=total_cls, initial=done_cls, desc="Classification tasks")
    try:
        for class_prompt in celeba_prompts:
            output["classification"]["celeba"].setdefault(class_prompt, {})
            for group in CELEBA_GROUPS:
                if has_classification_done(output, "celeba", class_prompt, group):
                    continue
                try:
                    t_g, variants = call_prompt_pair(
                        client=client,
                        t_in=class_prompt,
                        sensitive_attribute="intersection of age and gender",
                        group=group,
                        max_retries=args.max_retries,
                        sleep_sec=args.sleep_sec,
                    )
                except Exception as e:  # noqa: BLE001
                    err = str(e).strip() or repr(e)
                    if not is_empty_generation_error(err):
                        raise
                    failures.append(
                        {
                            "task_type": "classification",
                            "dataset": "celeba",
                            "class_prompt": class_prompt,
                            "sensitive_attribute": "intersection of age and gender",
                            "group": group,
                            "model": DEFAULT_GPT_MODEL,
                            "error": err,
                        }
                    )
                    save_failures_csv(failure_report_path, failures)
                    print(f"[Skipped] celeba | group={group} | reason={err}")
                    continue
                output["classification"]["celeba"][class_prompt][group] = {
                    "sensitive_attribute": "intersection of age and gender",
                    "t_g": t_g,
                    "T_g": variants,
                }
                save_json(output_path, output)
                cls_bar.update(1)
                cls_bar.set_postfix(model=DEFAULT_GPT_MODEL)
                print(f"[Saved] celeba | group={group} | model={DEFAULT_GPT_MODEL}")

        for class_prompt in facet_prompts:
            output["classification"]["facet"].setdefault(class_prompt, {})
            for group in FACET_GROUPS:
                if has_classification_done(output, "facet", class_prompt, group):
                    continue
                try:
                    t_g, variants = call_prompt_pair(
                        client=client,
                        t_in=class_prompt,
                        sensitive_attribute="gender",
                        group=group,
                        max_retries=args.max_retries,
                        sleep_sec=args.sleep_sec,
                    )
                except Exception as e:  # noqa: BLE001
                    err = str(e).strip() or repr(e)
                    if not is_empty_generation_error(err):
                        raise
                    failures.append(
                        {
                            "task_type": "classification",
                            "dataset": "facet",
                            "class_prompt": class_prompt,
                            "sensitive_attribute": "gender",
                            "group": group,
                            "model": DEFAULT_GPT_MODEL,
                            "error": err,
                        }
                    )
                    save_failures_csv(failure_report_path, failures)
                    print(f"[Skipped] facet | group={group} | reason={err}")
                    continue
                output["classification"]["facet"][class_prompt][group] = {
                    "sensitive_attribute": "gender",
                    "t_g": t_g,
                    "T_g": variants,
                }
                save_json(output_path, output)
                cls_bar.update(1)
                cls_bar.set_postfix(model=DEFAULT_GPT_MODEL)
                print(f"[Saved] facet | group={group} | model={DEFAULT_GPT_MODEL}")
    finally:
        cls_bar.close()

    print("Preparing retrieval tasks...")
    retrieval_datasets = cfg["retrieval"]["datasets"]
    retrieval_specs = {
        "flickr": ("perceived gender", FLICKR_GROUPS),
        "coco": ("intersection of perceived gender and skin tone", COCO_GROUPS),
    }

    rows_by_dataset: Dict[str, List[Dict[str, str]]] = {}
    total_ret = 0
    done_ret = 0
    for ds_name, ds_cfg in retrieval_datasets.items():
        rows = load_retrieval_rows(Path(ds_cfg["pairs_csv"]), max_rows=args.max_retrieval_rows)
        rows_by_dataset[ds_name] = rows
        groups = retrieval_specs[ds_name][1]
        total_ret += len(rows) * len(groups)
        for row in rows:
            for group in groups:
                probe = {
                    "dataset": ds_name,
                    "image_filename": row["image_filename"],
                    "text_query": row["text_query"],
                    "sensitive_attribute": retrieval_specs[ds_name][0],
                    "group": group,
                }
                if "split" in row:
                    probe["split"] = row["split"]
                if retrieval_key(probe) in existing_retrieval:
                    done_ret += 1
        print(f"Dataset={ds_name}: rows={len(rows)}, groups={len(groups)}")
    print(f"Retrieval progress: {done_ret}/{total_ret} already done.")

    ret_bar = tqdm(total=total_ret, initial=done_ret, desc="Retrieval tasks")
    try:
        for ds_name, rows in rows_by_dataset.items():
            sensitive_attribute, groups = retrieval_specs[ds_name]
            for row in rows:
                t_in = row["text_query"]
                for group in groups:
                    out_row = {
                        "dataset": ds_name,
                        "image_filename": row["image_filename"],
                        "text_query": t_in,
                        "sensitive_attribute": sensitive_attribute,
                        "group": group,
                    }
                    if "split" in row:
                        out_row["split"] = row["split"]
                    key = retrieval_key(out_row)
                    if key in existing_retrieval:
                        continue

                    try:
                        t_g, variants = call_prompt_pair(
                            client=client,
                            t_in=t_in,
                            sensitive_attribute=sensitive_attribute,
                            group=group,
                            max_retries=args.max_retries,
                            sleep_sec=args.sleep_sec,
                        )
                    except Exception as e:  # noqa: BLE001
                        err = str(e).strip() or repr(e)
                        if not is_empty_generation_error(err):
                            raise
                        failures.append(
                            {
                                "task_type": "retrieval",
                                "dataset": ds_name,
                                "image_filename": row["image_filename"],
                                "split": row.get("split", ""),
                                "text_query": t_in,
                                "sensitive_attribute": sensitive_attribute,
                                "group": group,
                                "model": DEFAULT_GPT_MODEL,
                                "error": err,
                            }
                        )
                        save_failures_csv(failure_report_path, failures)
                        print(
                            "[Skipped] "
                            f"{ds_name} | image={row['image_filename']} | group={group} | reason={err}"
                        )
                        continue
                    out_row["t_g"] = t_g
                    out_row["T_g"] = variants
                    if key in retrieval_row_index:
                        output["retrieval"][retrieval_row_index[key]] = out_row
                    else:
                        output["retrieval"].append(out_row)
                        retrieval_row_index[key] = len(output["retrieval"]) - 1
                    existing_retrieval.add(key)
                    save_json(output_path, output)
                    ret_bar.update(1)
                    ret_bar.set_postfix(model=DEFAULT_GPT_MODEL, dataset=ds_name)
                    print(
                        "[Saved] "
                        f"{ds_name} | image={row['image_filename']} | group={group} | model={DEFAULT_GPT_MODEL}"
                    )
    finally:
        ret_bar.close()

    print("Preparing generation tasks...")
    generation_specs: List[Dict[str, str]] = []
    for p in NEUTRAL_GENERATION_PROMPTS:
        for g in GENERATION_GROUPS:
            generation_specs.append(
                {
                    "dataset": "generation",
                    "prompt_set": "neutral",
                    "text_query": p,
                    "sensitive_attribute": "perceived gender",
                    "group": g,
                }
            )
    for p in GROUP_EXPLICIT_GENERATION_PROMPTS:
        generation_specs.append(
            {
                "dataset": "generation",
                "prompt_set": "group_explicit",
                "text_query": p,
                "sensitive_attribute": "perceived gender",
                "group": detect_group_from_explicit_prompt(p),
            }
        )

    total_gen = len(generation_specs)
    done_gen = 0
    for row in generation_specs:
        if generation_key(row) in existing_generation:
            done_gen += 1
    print(f"Generation progress: {done_gen}/{total_gen} already done.")

    gen_bar = tqdm(total=total_gen, initial=done_gen, desc="Generation tasks")
    try:
        for row in generation_specs:
            key = generation_key(row)
            if key in existing_generation:
                continue

            try:
                t_g, variants = call_prompt_pair(
                    client=client,
                    t_in=row["text_query"],
                    sensitive_attribute=row["sensitive_attribute"],
                    group=row["group"],
                    max_retries=args.max_retries,
                    sleep_sec=args.sleep_sec,
                )
            except Exception as e:  # noqa: BLE001
                err = str(e).strip() or repr(e)
                if not is_empty_generation_error(err):
                    raise
                failures.append(
                    {
                        "task_type": "generation",
                        "dataset": "generation",
                        "prompt_set": row["prompt_set"],
                        "text_query": row["text_query"],
                        "sensitive_attribute": row["sensitive_attribute"],
                        "group": row["group"],
                        "model": DEFAULT_GPT_MODEL,
                        "error": err,
                    }
                )
                save_failures_csv(failure_report_path, failures)
                print(
                    "[Skipped] "
                    f"generation | set={row['prompt_set']} | group={row['group']} | reason={err}"
                )
                continue

            out_row = {
                "dataset": "generation",
                "prompt_set": row["prompt_set"],
                "text_query": row["text_query"],
                "sensitive_attribute": row["sensitive_attribute"],
                "group": row["group"],
                "t_g": t_g,
                "T_g": variants,
            }
            if key in generation_row_index:
                output["generation"][generation_row_index[key]] = out_row
            else:
                output["generation"].append(out_row)
                generation_row_index[key] = len(output["generation"]) - 1
            existing_generation.add(key)
            save_json(output_path, output)
            gen_bar.update(1)
            gen_bar.set_postfix(model=DEFAULT_GPT_MODEL)
            print(
                "[Saved] "
                f"generation | set={row['prompt_set']} | group={row['group']} | model={DEFAULT_GPT_MODEL}"
            )
    finally:
        gen_bar.close()

    save_json(output_path, output)
    print(f"Saved results to: {output_path}")
    if failures:
        print(f"Saved failure report to: {failure_report_path} (rows={len(failures)})")
    print(
        "Summary: "
        f"classification_celeba_prompts={len(output['classification']['celeba'])}, "
        f"classification_facet_prompts={len(output['classification']['facet'])}, "
        f"retrieval_rows={len(output['retrieval'])}, "
        f"generation_rows={len(output['generation'])}"
    )


if __name__ == "__main__":
    main()

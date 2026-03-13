# Reproduction Guide

## 1. Install dependencies


```bash
python -m pip install -r requirements.txt
```

## 2. Prepare the datasets

Download the datasets from their official sources:

- CelebA
- COCO 2017
- Flickr30k
- FACET

After downloading:

1. Construct the evaluation subsets as described in the appendix.
2. Update the paths in `config.yaml` so they point to your local copies.

## 3. Construct `LLM.json`

Run `gpt.py` to generate the prompt expansions used by the experiments.

```bash
python gpt.py --output_json /your/path/LLM.json
```

Then update `paths.llm_json` in `config.yaml`.

## 4. Run classification

Example commands:

```bash
python classification.py --dataset celeba
python classification.py --dataset facet
```

To enable debiasing:

```bash
python classification.py --dataset celeba --apply_debiasing
python classification.py --dataset facet --apply_debiasing
```



Example with explicit model selection:

```bash
python classification.py --dataset celeba --models clip_vit_l14 clip_rn50 blip
```

## 5. Run retrieval

Example commands:

```bash
python retrieve.py --dataset coco
python retrieve.py --dataset flickr
```

To enable debiasing:

```bash
python retrieve.py --dataset coco --apply_debiasing
python retrieve.py --dataset flickr --apply_debiasing
```

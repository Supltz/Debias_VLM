<div align="center">

## A Closed-Form Solution for Debiasing Vision-Language Models<br>with Utility Guarantees Across Modalities and Tasks

**CVPR 2026 | Tangzheng Lian, Guanyu Hu, Yijing Ren, Dimitrios Kollias, Oya Celiktutan**

[![arXiv](https://img.shields.io/badge/arXiv-2603.12998-b31b1b)](https://arxiv.org/pdf/2603.12998)
[![Paper](https://img.shields.io/badge/Paper-CVF-4d4d4d)](https://openaccess.thecvf.com/content/CVPR2026/papers/Lian_A_Closed-Form_Solution_for_Debiasing_Vision-Language_Models_with_Utility_Guarantees_CVPR_2026_paper.pdf)
[![License](https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11.3-1f77b4)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-ee4c2c)

</div>

---


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


## Citation

Please cite our work if you use this code:

```bibtex
@inproceedings{lian2026closed,
  title={A closed-form solution for debiasing vision-language models with utility guarantees across modalities and tasks},
  author={Lian, Tangzheng and Hu, Guanyu and Ren, Yijing and Kollias, Dimitrios and Celiktutan, Oya},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={31672--31682},
  year={2026}
}
```

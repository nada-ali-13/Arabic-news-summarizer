# Asset verification and technical decisions

Checked 2026-11-06 using the public upstream pages, without downloading full weights.

| Asset | Evidence | Decision |
|---|---|---|
| `UBC-NLP/AraT5v2-base-1024` | [Model card](https://huggingface.co/UBC-NLP/AraT5v2-base-1024), [config](https://huggingface.co/UBC-NLP/AraT5v2-base-1024/blob/main/config.json) | `model_type=t5`, encoder-decoder, T5Tokenizer; compatible with AutoModelForSeq2SeqLM. Card describes 1,024-token context. Keep requested model. |
| Model file format | [Official file listing](https://huggingface.co/UBC-NLP/AraT5v2-base-1024/tree/main) | Published `pytorch_model.bin` about 1.47 GB, no `model.safetensors` listed. Load pinned original state dictionary using PyTorch ≥2.6, `weights_only=True`, `use_safetensors=False`. Export trained weights as safetensors. Do not rely on an account-dependent conversion service. |
| Model weight license | [Raw exact model card](https://huggingface.co/UBC-NLP/AraT5v2-base-1024/raw/main/README.md), [upstream repository](https://github.com/UBC-NLP/araT5) | Exact card has no license field and model file listing has no LICENSE. Upstream describes research availability, but that is not an explicit weight redistribution license. Public download verified; permission for weight redistribution remains unresolved. No license is invented or inherited from unrelated models. |
| `csebuetnlp/xlsum`, Arabic | [Dataset card](https://huggingface.co/datasets/csebuetnlp/xlsum), [original loader](https://huggingface.co/datasets/csebuetnlp/xlsum/blob/main/xlsum.py) | Fields id/url/title/summary/text, official train/val/test. Arabic counts 37,519 / 4,689 / 4,689 before cleaning. Keep requested dataset. |
| Dataset loading | [Official archive listing](https://huggingface.co/datasets/csebuetnlp/xlsum/tree/main/data) | Arabic v2.0 archive is 44.9 MB. Modern datasets removed dataset-script execution; read the identical official archive via hf_hub_download and construct DatasetDict locally, without trust_remote_code. |
| Dataset license | [Dataset card licensing](https://huggingface.co/datasets/csebuetnlp/xlsum#licensing-information) | CC BY-NC-SA 4.0. Retain attribution to XL-Sum authors/BBC sources and comply with noncommercial/share-alike terms. Dataset and model rights are separate. |

Runtime resolves model/dataset revisions to immutable commit SHAs in `run_config.json`; the notebook also freezes source commit. Availability was checked via public metadata, not a full model download/tokenizer smoke in this build environment. Colab must verify actual downloads, tokenizer behavior, T4 memory and numerical stability.

Selected API versions: Transformers 4.57.1, Accelerate 1.11.0, Datasets 4.3.0, Gradio 5.49.1, SentencePiece 0.2.1, Trafilatura 2.0.0, PyTorch ≥2.6,<3. CPU integration uses PyTorch 2.8.0+cpu and Python 3.12.14. Transitive versions are recorded by the notebook; broad non-API-sensitive constraints may resolve differently over time.

Cite the original research:

- Nagoudi et al., [AraT5: Text-to-Text Transformers for Arabic Language Generation](https://aclanthology.org/2022.acl-long.47/), ACL 2022.
- Elmadany et al., [Octopus: A Multitask Model and Toolkit for Arabic Natural Language Generation](https://aclanthology.org/2023.arabicnlp-1.20/), ArabicNLP 2023.
- Hasan et al., [XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages](https://aclanthology.org/2021.findings-acl.413/), Findings ACL 2021.

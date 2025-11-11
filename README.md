<div align="center">

# 🩺 PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-red.svg)](https://aaai.org/Conferences/AAAI-26/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.05353-b31b1b.svg)](https://arxiv.org/abs/2508.05353)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/MK-runner/PriorRG)
[![BibTeX](https://img.shields.io/badge/%F0%9F%93%96-BibTeX-yellow)](#-citation)

<img src="generated_reports/figure2.png" alt="Framework Overview" width="75%">

</div>

---

## 📰 News

- **[2025-11-10]** Released [**generated reports**](https://github.com/mk-runner/PriorRG/blob/main/generated_reports/mimic-cxr-generated-reports-24-03-2025_18-07-41.csv) → `reference_report` = ground truth, `generated_report` = model output  
- **[2025-11-10]** Official code and [pre-trained weights](https://huggingface.co/MK-runner/PriorRG) are now public.

---

## ⚙️ Installation

```bash
# Create environment
conda create -n priorrg python=3.9.0
conda activate priorrg

# Install dependencies
pip install -r requirements.txt
````

**Core dependencies:**

* `transformers==4.43.3`
* `radgraph==0.09`

> See `requirements.txt` for the complete list of dependencies.

---

## 🧩 Model Checkpoints

| Dataset       | Checkpoints                                                                             | Generated Reports                                                                                                           |
| ------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **MIMIC-CXR** | [HuggingFace](https://huggingface.co/MK-runner/PriorRG/tree/main/checkpoints/mimic-cxr) | [CSV](https://github.com/mk-runner/PriorRG/blob/main/generated_reports/mimic-cxr-generated-reports-24-03-2025_18-07-41.csv) |
| **MIMIC-ABN** | [HuggingFace](https://huggingface.co/MK-runner/PriorRG/tree/main/checkpoints/mimic-cxr) | [CSV](https://github.com/mk-runner/PriorRG/blob/main/generated_reports/mimic-cxr-generated-reports-24-03-2025_18-07-41.csv) |

---

## 📁 Dataset Structure

### 1. Medical Images

PriorRG is trained on **MIMIC-CXR** and **MIMIC-ABN** datasets from [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/).

```
data/
├── p10/
│   └── p10000032/
│       └── s50414267/
│           ├── 02aa804e-....jpg
│           └── 174413ec-....jpg
├── p11/
└── ...
```

### 2. Radiology Reports

Organized by `study_id` to obtain longitudinal data.

| Dataset            | Processed File                                                                                                                                                      | Description                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| **MIMIC-CXR**      | [`priorrg_mimic_cxr_annotation.json`](https://huggingface.co/MK-runner/PriorRG/blob/main/checkpoints/mimic-cxr/radiology-reports/priorrg_mimic_cxr_annotation.json) | Report annotations for MIMIC-CXR  |
| **MIMIC-ABN**      | [`priorrg_mimic_abn_annotation.json`](https://huggingface.co/MK-runner/PriorRG/blob/main/checkpoints/mimic-cxr/radiology-reports/priorrg_mimic_abn_annotation.json) | Report annotations for MIMIC-ABN  |
| **View Positions** | [`view_position_dict.json`](https://huggingface.co/MK-runner/PriorRG/blob/main/checkpoints/mimic-cxr/radiology-reports/view-positions-dict-mimic.json)              | Metadata for X-ray view positions |

### 3. Checkpoint Directory Layout

```
ckpt_zoo_dir/
├── chexbert.pth
├── radgraph/
├── google-bert/bert-base-uncased/
├── microsoft/BiomedVLP-CXR-BERT-specialized/
├── microsoft/rad-dino/
└── distilbert/distilgpt2/
```

> `chexbert.pth` and `radgraph` must be downloaded manually (see [MLRG](https://github.com/mk-runner/MLRG) for instructions).
> Other checkpoints will be automatically fetched during training.

---

## 🚀 Inference

The script `main_single_sample_github.py` supports **four input configurations** for single-study inference:

| Input Type                | Description                                                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🩻 **Image only**         | Single X-ray without view position (`view_position='unk'`)                                                                                                                      |
| 🧭 **+ View position**    | Specify position (e.g., PA, AP, Lateral). See [`view_position_dict.json`](https://huggingface.co/MK-runner/PriorRG/blob/main/radiology-report/priorrg_view_position_v1.0.json). |
| 💬 **+ Clinical context** | Add optional clinical notes or findings                                                                                                                                         |
| 📜 **+ Prior study**      | Provide a previous X-ray for longitudinal reasoning                                                                                                                             |

> Example configurations are available in `main_single_sample_github.py`.

---

## 🧠 Training & Evaluation Pipeline (MIMIC-CXR)

```bash
# Pretraining (finetune mode)
bash script_github/mimic-cxr-pretraining-finetune.sh

# Pretraining (inference mode)
bash script_github/mimic-cxr-pretraining-inference.sh

# Report generation (finetune mode)
bash script_github/mimic-cxr-report-generation-finetune.sh

# Report generation (inference mode)
bash script_github/mimic-cxr-report-generation-inference.sh
```

---

## 📊 Evaluation

```python
def compute_performance_using_generated_reports():
    from tools.metrics.metrics import compute_all_scores, compute_chexbert_details_scores
    import pandas as pd

    mimic_cxr_generated_path = 'generated_reports/mimic-cxr-generated-reports-24-03-2025_18-07-41.csv'
    args = {
        'chexbert_path': "/home/miao/data/dataset/checkpoints/chexbert.pth",
        'bert_path': "/home/miao/data/dataset/checkpoints/bert-base-uncased",
        'radgraph_path': "/home/miao/data/dataset/checkpoints/radgraph",
    }

    data = pd.read_csv(mimic_cxr_generated_path)
    gts, gens = data['reference_report'].tolist(), data['generated_report'].tolist()
    scores = compute_all_scores(gts, gens, args)
    print(scores)
```

---

## 📚 Citation

If you find this work helpful, please cite:

```bibtex
@misc{liu2025priorrgpriorguidedcontrastivepretraining,
  title={PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation},
  author={Kang Liu and Zhuoqi Ma and Zikang Fang and Yunan Li and Kun Xie and Qiguang Miao},
  year={2025},
  eprint={2508.05353},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.05353}
}
```

---

## 🙏 Acknowledgements

* [MLRG](https://github.com/mk-runner/MLRG): Dataset organization and evaluation tools
* [cvt2distilgpt2](https://github.com/aehrc/cvt2distilgpt2): Text generation initialization framework

---

<div align="center">

⭐️ **If you find this repository useful, please consider starring it!**
📬 For questions, open an issue or contact the authors.

</div>

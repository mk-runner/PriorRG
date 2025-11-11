<div align="center">

# PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-red.svg)](https://aaai.org/Conferences/AAAI-26/)&nbsp;&nbsp;&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2508.05353-b31b1b.svg)](https://arxiv.org/abs/2508.05353)&nbsp;&nbsp;&nbsp;
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/MK-runner/PriorRG)&nbsp;&nbsp;&nbsp;
[![BibTeX](https://img.shields.io/badge/%F0%9F%93%96-BibTeX-yellow)](#-Citation)

<div align="center">
  <img src="generated_reports/figure2.png" alt="Framework" width="75%">
</div>

</div>

---

## 📢 News

- **2025-11-10** &nbsp; Upload [**generated reports**](https://github.com/mk-runner/PriorRG/blob/main/generated_reports/mimic-cxr-generated-reports-24-03-2025_18-07-41.csv) — `reference_reports` = ground-truth reports, `report` = generated reports  
- **2025-11-10** &nbsp; Public release of official code & [pre-trained weights](https://huggingface.co/MK-runner/PriorRG)

---

## ⚙️ Installation

```bash
# create environment
conda create -n priorrg python=3.9.0

# install dependencies
pip install -r requirements.txt
```
**Core dependencies**

* `transformers==4.43.3` 
* `radgraph==0.09`
  
> Please see `requirements.txt` for additional dependencies.

---


## 🧩 Model Checkpoints

| Dataset       | Download                                                                    | Generated Reports                                                               |
| ------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **MIMIC-CXR** | [HuggingFace](https://huggingface.co/MK-runner/PriorRG/tree/main/checkpoints/mimic-cxr) | [CSV](https://github.com/mk-runner/PriorRG/blob/main/generated_reports/mimic-cxr-generated-reports-24-03-2025_18-07-41.csv) |
| **MIMIC-ABN** | [HuggingFace](https://huggingface.co/MK-runner/PriorRG/tree/main/checkpoints/mimic-cxr) | [CSV](https://github.com/mk-runner/PriorRG/blob/main/generated_reports/mimic-cxr-generated-reports-24-03-2025_18-07-41.csv) |

---

## 📂 Dataset Structure

### 1. Medical Images

PriorRG uses **MIMIC-CXR** and **MIMIC-ABN** datasets from [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/).

```
data/
├── p10
│   └── p10000032
│       └── s50414267
│           ├── 02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
│           └── 174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg
├── p11
├── ...
└── p19
```

### 2. Radiology Reports

Organized by `study_id` to align longitudinal data.

| Dataset        | Processed File                                                                                                                                         | Description                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| MIMIC-CXR      | [`priorrg_mimic_cxr_annotation.json`](https://huggingface.co/MK-runner/PriorRG/blob/main/checkpoints/mimic-cxr/radiology-reports/priorrg_mimic_cxr_annotation.json) | annotation for the MIMIC-CXR dataset         |
| MIMIC-ABN      | [`priorrg_mimic_abn_annotation.json`](https://huggingface.co/MK-runner/PriorRG/blob/main/checkpoints/mimic-cxr/radiology-reports/priorrg_mimic_abn_annotation.json) | annotation for the MIMIC-ABN dataset |
| View Positions | [`view_position_dict.json`](https://huggingface.co/MK-runner/PriorRG/blob/main/checkpoints/mimic-cxr/radiology-reports/view-positions-dict-mimic.json)               | View position metadata for all studies           |

### 3. `ckpt_zoo_dir` in `tool/utils_github.py`
```
'ckpt_zoo_dir'/
├── chexbert.pth
├── radgraph
├── google-bert/bert-base-uncased
├── microsoft/BiomedVLP-CXR-BERT-specialized
├── microsoft/rad-dino
└── distilbert/distilgpt2
```

> `chexbert.pth` and `radgraph` used for evaluation metrics need to be downloaded manually, while other checkpoints are automatically downloaded by the code.

> The download instructions for `chexbert.pth` and `radgraph` follow those provided in [MLRG](https://github.com/mk-runner/MLRG).

---

## 🚀 Usage — Inference with `main_single_sample_github.py`

The script `main_single_sample_github.py` allows single-study inference under **four input configurations**:

| Input Type                           | Description                                        |
| ------------------------------------ | -------------------------------------------------- |
| 🩻 **Image only**                    | A single image without view position (i.e., `view_position = 'unk'`)             |
| 🧭 **+ View position**               | Specify the view position (e.g., PA, AP, Lateral, for more details see [`view_position_dict.json`](https://huggingface.co/MK-runner/PriorRG/blob/main/radiology-report/priorrg_view_position_v1.0.json))  |
| 💬 **+ Clinical context (optional)** | Provide brief clinical notes or findings           |
| 📜 **+ Prior study (optional)**      | Supply a previous X-ray for longitudinal reasoning |

> Examples can be found in `main_single_sample_github.py`



---

## 🧠 Pipeline on the MIMIC-CXR Dataset

```bash
# conduct pretraining task (finetune mode) on the MIMIC-CXR dataset
bash script_github/mimic-cxr-pretraining-finetune.sh

# conduct pretraining task (inference mode) on the MIMIC-CXR dataset
bash script_github/mimic-cxr-pretraining-inference.sh

# conduct report generation task (finetune mode) on the MIMIC-CXR dataset
bash script_github/mimic-cxr-report-generation-finetune.sh

# conduct report generation task (inference mode) on the MIMIC-CXR dataset
bash script_github/mimic-cxr-report-generation-inference.sh

```

## 📊 Evaluation using generated radiology reports

```python
def compute_performance_using_generated_reports():
    from tools.metrics.metrics import compute_all_scores, compute_chexbert_details_scores
    mimic_cxr_generated_path = 'generated_reports/mimic-cxr-generated-reports-24-03-2025_18-07-41.csv'
    #mimic_abn_generated_path = 'generated-radiology-reports/MIMIC-ABN/test_reports_epoch-1_23-10-2024_10-25-20.csv'
    args = {
        'chexbert_path': "/home/miao/data/dataset/checkpoints/chexbert.pth",
        'bert_path': "/home/miao/data/dataset/checkpoints/bert-base-uncased",
        'radgraph_path': "/home/miao/data/dataset/checkpoints/radgraph",
    }
    for generated_path in [mimic_cxr_generated_path, mimic_abn_generated_path, twoview_cxr_generated_path]:
        data = pd.read_csv(generated_path)
        gts, gens = data['reference_report'].tolist(), data['generated_report'].tolist()
        scores = compute_all_scores(gts, gens, args)
        print(scores)
```

---

## 📜 Citation

If you use or extend our work, please cite our AAAI 2026 paper.

```bibtex
@misc{liu2025priorrgpriorguidedcontrastivepretraining,
      title={PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation}, 
      author={Kang Liu and Zhuoqi Ma and Zikang Fang and Yunan Li and Kun Xie and Qiguang Miao},
      year={2025},
      eprint={2508.05353},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.05353}, 
}
```
---

## 🙏 Acknowledgements

* [MLRG](https://github.com/mk-runner/MLRG) — foundational dataset organization
* [cvt2distilgpt2](https://github.com/aehrc/cvt2distilgpt2) — text generation initialization

---

<div align="center">

⭐️ If you find this repository useful, please consider starring it!
📬 For questions, open an issue or contact the authors.

</div>

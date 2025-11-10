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

## ⚙️ Requirements

```bash
# create environment
conda create -n priorrg python=3.9.0

# install dependencies
pip install -r requirements.txt
```
## Usage

```bash
# conduct pretraining task on the MIMIC-CXR dataset


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

---

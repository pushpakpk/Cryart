# PyTorch Week 3 — ResNet-18 & Transformer Implementation

## Overview
This project implements two classic deep learning architectures **from scratch** using PyTorch:

1. **ResNet-18** — CIFAR-10 image classification  
2. **Transformer Encoder-Decoder** — Toy sequence-to-sequence translation

_No high-level prebuilt modules (`torchvision.models` or `nn.Transformer`) are used._

---

## Repo Structure

```
code/
├─ cls/    # ResNet-18 code: dataset, model, train, eval, gradcam
├─ mt/     # Transformer code: dataset, model, train, eval_decode
└─ utils/  # Metrics and visualization helpers

runs/
├─ cls/    # ResNet visualizations: curves, confusion, Grad-CAM
└─ mt/     # Transformer visualizations: curves, attention, BLEU

report/
└─ one_page_report.md

README.md
requirements.txt
```

---

## Setup

1. Clone the repo:
```bash
git clone <repo_url>
cd pytorch-week3
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run training and evaluation as shown in **Usage** below.

---

## Usage

### ResNet-18
```bash
python code/cls/train.py        # Train model
python code/cls/eval.py         # Confusion matrix & prediction grids
python code/cls/gradcam.py      # Grad-CAM visualization
```

### Transformer
```bash
python code/mt/train.py         # Train Transformer
python code/mt/eval_decode.py   # Generate translations & compute BLEU
```

---

## Results

- **ResNet-18:** Test accuracy ≥ 80%, clear diagonal confusion matrix, Grad-CAM highlights.  
- **Transformer:** BLEU score ≥ 15, interpretable attention heatmaps, coherent generated sequences.

---

## References

- He et al., 2015. *Deep Residual Learning for Image Recognition.* https://arxiv.org/abs/1512.03385  
- Vaswani et al., 2017. *Attention Is All You Need.* https://arxiv.org/abs/1706.03762  
- GeeksforGeeks. CIFAR-10 Dataset Loader.  
- Selvaraju et al., 2016. *Grad-CAM.* https://arxiv.org/abs/1610.02391  
- Papineni et al., 2002. *BLEU.* https://aclanthology.org/P02-1040/

---

## Notes

- Implementations avoid high-level prebuilt modules to teach fundamentals.  
- Check `runs/` for visualizations and training artifacts.  
- Set correct CUDA device or use CPU by adjusting training scripts.

---

## Expected Outputs

- Learning curves (loss & accuracy)  
- Confusion matrix and Grad-CAM images (ResNet)  
- Attention heatmaps and BLEU report (Transformer)

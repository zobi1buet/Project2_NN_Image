# Project 2 — Veggie Classification with PyTorch

Classify images of vegetables into 10 categories using PyTorch. Build a CNN from scratch and fine-tune a pretrained model.

## Dataset

**Vegetables** — 24,000 images across 10 vegetable classes.
- Training: 20,000 images
- Validation: 4,000 images

Download `Vegetables.zip` from Brightspace and place it in your working directory.

## Files

```
.
├── project2_starter.ipynb     # Student assignment notebook
├── utils.py                   # Shared helper functions
├── requirements.txt           # Dependencies
└── README.md
```

## Getting Started on Google Colab

1. Upload `Vegetables.zip` and the project files to Colab
2. Go to **Runtime > Change runtime type > GPU** (T4 is fine)
3. Run all cells — the zip extracts automatically

## Deliverables

- Your completed github repository link
- Your saved model file (`.pth`) — must be under 400 MB
- A note (~1-2 paragraphs) explaining what you did to improve accuracy

## Grading

| Component | Weight |
|---|---|
| Accuracy (validation set) | 60% |
| Code — readable and logical | 20% |
| Explanatory note | 20% |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib, numpy

All are pre-installed on Google Colab.

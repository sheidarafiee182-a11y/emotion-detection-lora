# 🧠 Multi-Label Emotion Detection with RoBERTa + LoRA

**Fine-tuning a RoBERTa-base model for multi-label emotion detection on the GoEmotions dataset using Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA).**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

This project fine-tunes a **RoBERTa-base** model for **multi-label emotion detection** on the **GoEmotions** dataset using **Parameter-Efficient Fine-Tuning (PEFT)** with **Low-Rank Adaptation (LoRA)**.

The model classifies text into **28 distinct emotions**, with a special focus on handling **imbalanced classes** (e.g., `grief` with only 77 samples vs `neutral` with 14,219 samples).

### 🎯 Key Features
- ✅ **28 emotion classes** from the GoEmotions dataset
- ✅ **Parameter-Efficient** training with LoRA (< 1% trainable parameters)
- ✅ **Handling class imbalance** with Focal Loss and Class Weights
- ✅ **Lightweight adapter** (only ~5MB vs 500MB full model)
- ✅ **Augmented dataset** for rare classes (target: 600 samples per class)
- ✅ **Comprehensive evaluation** with F1-Macro, F1-Micro, Accuracy, Confusion Matrices

---

## 📊 Dataset

### Class Distribution (Original)

![Original Distribution](images/original_distribution.png)

| Class | Samples | Weight |
|-------|---------|--------|
| grief (16) | 77 | 9.39 |
| nervousness (19) | 437 | 4.21 |
| pride (21) | 282 | 6.54 |
| relief (23) | 388 | 4.75 |
| neutral (27) | 14,219 | 0.13 |

### Augmented Dataset

![Augmentation Comparison](images/augmentation_comparison.png)

---

## 🛠️ Approach

### Why LoRA?

| Metric | Full Fine-Tuning | LoRA (PEFT) |
|--------|------------------|-------------|
| **Trainable Parameters** | ~125M | **< 1%** (~1.5M) |
| **Model Size** | ~500MB | **~5-10MB** |
| **GPU Memory** | High (12-16GB) | **Low (4-8GB)** |

### Model Architecture

![Model Architecture](images/model_architecture_block_diagram.png)

---

## 📈 Results

### Model Comparison

![Model Comparison](images/model_comparison.png)

| Metric | Model A (Advanced) | Model B (Baseline) | Improvement |
|--------|-------------------|-------------------|-------------|
| **F1-Macro** | **0.236** | 0.182 | **+29.7%** ✅ |
| **F1-Micro** | **0.460** | 0.340 | **+35.3%** ✅ |
| **Accuracy** | **0.319** | 0.211 | **+51.2%** ✅ |

### Confusion Matrices

#### Model A (Advanced)
![Confusion Matrix A](images/confusion_matrix_A.png)

#### Model B (Baseline)
![Confusion Matrix B](images/confusion_matrix_B.png)

---

## 📂 Project Structure
emotion-detection-lora/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│ ├── tokenized.zip
│ └── tokenized_augmented/
├── models/
│ └── best-model/
├── src/
│ └── train_lora.py
├── notebooks/
│ ├── final_lora.ipynb
│ └── preprocessing.ipynb
├── images/
│ ├── original_distribution.png
│ ├── augmentation_comparison.png
│ ├── model_comparison.png
│ ├── confusion_matrix_A.png
│ └── confusion_matrix_B.png
└── results/
├── test_results_A.json
└── test_results_B.json


---

## 💻 Installation

```bash
git clone https://github.com/sheidarafiee182-a11y/emotion-detection-lora.git
cd emotion-detection-lora
pip install -r requirements.txt

🚀 Usage
python
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from peft import PeftModel

base_model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=28,
    ignore_mismatched_sizes=True
)

model = PeftModel.from_pretrained(base_model, "models/best-model")
tokenizer = RobertaTokenizer.from_pretrained("models/best-model")

text = "I absolutely love this movie!"
predictions = predict_emotion(text, model, tokenizer)
👨‍💻 Author
Sheida Rafiee

GitHub: @sheidarafiee182-a11y


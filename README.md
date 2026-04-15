# 📩 SMS Spam Detection using NLP

A Natural Language Processing project that detects spam SMS messages using Naive Bayes and Support Vector Machine (SVM) classifiers.

---

## 📋 Overview

This project builds and compares two NLP classification pipelines to distinguish between spam and legitimate (ham) SMS messages. Text messages are preprocessed, converted into numerical features using TF-IDF, and then classified using Multinomial Naive Bayes and a linear SVM.

---

## 🎯 Objectives

- Preprocess raw SMS text data for machine learning
- Visualize the distribution of spam vs. ham messages
- Train and evaluate Naive Bayes and SVM classifiers
- Compare model performance using accuracy, confusion matrix, and classification report

---

## 📁 Repository Structure

```
├── NLP.py        # Main Python script
├── spam.csv      # Dataset
└── README.md
```

---

## 📊 Dataset

The dataset (`spam.csv`) is included in this repository.

**Original source:** [SMS Spam Collection Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

**Key columns:**
| Column | Description |
|---|---|
| `Label` | `ham` (legitimate) or `spam` |
| `Message` | Raw SMS text content |

---

## 🔍 Methodology

### Text Preprocessing
Each SMS message goes through the following steps:
- Lowercasing
- Tokenization
- Punctuation removal
- Stopword removal
- Lemmatization (using WordNet)

### Feature Extraction
- **TF-IDF Vectorization** — converts preprocessed text into numerical feature matrices, weighing words by their importance across the corpus

### Models

| Model | Description |
|---|---|
| **Multinomial Naive Bayes** | Probabilistic classifier well-suited for word count / TF-IDF features |
| **SVM (Linear Kernel)** | Finds the optimal hyperplane to separate spam from ham with maximum margin |

Both models are implemented as **scikit-learn Pipelines** (TF-IDF → Classifier).

---

## 📈 Evaluation Metrics

- **Accuracy** — overall correct predictions
- **Confusion Matrix** — true/false positives and negatives
- **Classification Report** — precision, recall, and F1-score per class

---

## 🛠️ Installation & Usage

### Requirements
```bash
pip install pandas nltk scikit-learn matplotlib seaborn
```

Also download the required NLTK resources (handled automatically in the script):
```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Run the Script
```bash
python NLP.py
```

---

## 📌 Key Findings

- Ham messages significantly outnumber spam messages in the dataset (class imbalance)
- Both Naive Bayes and SVM perform well on SMS spam detection with TF-IDF features
- The linear SVM is particularly effective due to the high-dimensional, sparse nature of text data

---

## 📚 References

- Dataset: [SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [NLTK Documentation](https://www.nltk.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)

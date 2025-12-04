This repository contains the source code, datasets, and experimental results for the research project thesis: **"Evaluating Cross-Domain Adaptability of Transformer Models: A Comparative Analysis of Social Media and Product Review Sentiments."**
The project systematically evaluates how well six state-of-the-art transformer models generalize sentiment analysis tasks across fundamentally different text domains: informal social media (Twitter, Reddit) versus structured product reviews (Amazon, IMDb).

### **Research Goal**
To quantify the "domain gap" in sentiment analysis—the performance drop when a model trained on one type of text is applied to another without additional fine-tuning—and to analyze the trade-offs between model accuracy and computational efficiency.

### **Models Evaluated**
- **BERT** (bert-base-uncased)
- **RoBERTa** (roberta-base)
- **ELECTRA** (electra-base-discriminator)
- **DistilBERT** (distilbert-base-uncased)
- **ALBERT** (albert-base-v2)
- **XLNet** (xlnet-base-cased)

### **Datasets**
Four benchmark datasets representing two distinct domains:
1. **Social Media:** Sentiment140 (Twitter), Reddit Comments
2. **Product Reviews:** Amazon Reviews, IMDb Movie Reviews

## Repository Structure

```
project-thesis-nlp-xdomain/
│
├── data/                        # Raw and processed datasets
│   ├── sentiment140/            # Twitter data
│   ├── reddit/                  # Reddit comments
│   ├── amazon/                  # Amazon product reviews
│   └── imdb/                    # IMDb movie reviews
│
├── code/                        # Core Python scripts for training & eval
│   ├── main.py                  # Entry point for running experiments
│   ├── train_transformers.py    # Training pipeline using Hugging Face Trainer
│   ├── data_processors.py       # Data loading and tokenization logic
│   ├── config.py                # Hyperparameters and configuration
│   ├── preprocessors.py         # Domain-specific cleaning functions
│   └── sentiment_mappers.py     # Label mapping (Binary/Ternary standardization)
│
├── outputs/                     # Experimental results & logs
│   ├── metrics_summary.csv      # In-domain performance metrics (Table 1-5 in thesis)
│   ├── cross_domain_results.csv # Cross-domain transfer metrics (Table 6-11 in thesis)
    ├── training_log.txt         # Complete training logs
    ├── performance_gap.jpg      # In-domain vs Cross-domain comparison
    ├── transfer_heatmap.jpg     # Heatmap of cross-domain performance
    └── efficiency_tradeoff.jpg  # Accuracy vs Training Time bubble chart
```

## Getting Started

### **1. Prerequisites**
- Python 3.8+
- CUDA-enabled GPU (Recommended: RTX 3060/4060/5060 or better for FP16 training)

### **2. Installation**
Clone the repository and install dependencies:

```bash
git clone https://github.com/sraboni-bhuiyan/project-thesis-nlp-xdomain.git
cd project-thesis-nlp-xdomain
pip install -r requirements.txt
```

*Note: Key dependencies include `torch`, `transformers`, `scikit-learn`, `pandas`, and `numpy`.*


## Key Results

| Metric | Best Model (In-Domain) | Best Model (Cross-Domain) | Fastest Model |
|:-------|:----------------------:|:-------------------------:|:-------------:|
| **Model** | **RoBERTa** | **BERT** | **DistilBERT** |
| **Score** | 0.7424 Macro-F1 | 0.4083 Macro-F1 | 135s / dataset |

- **Domain Gap:** Average performance drop of **~45%** when moving from in-domain to cross-domain.
- **Hardest Dataset:** **Reddit** proved most challenging due to informal language and neutral class ambiguity.
- **Efficiency:** **DistilBERT** offered 3x speedup over XLNet with only marginal accuracy loss.

---

## Citation & References

If you use this code or findings, please reference the thesis project:

> Bhuiyan, S. (2025). *Evaluating Cross-Domain Adaptability of Transformer Models*. Master's Thesis, Fachhochschule Dortmund.

---

## Author

**Sraboni Bhuiyan**  
Master Digital Transformation | Fachhochschule Dortmund  
[GitHub Profile](https://github.com/sraboni-bhuiyan)


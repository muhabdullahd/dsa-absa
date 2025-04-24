# Aspect-Based Sentiment Analysis (ABSA) for Restaurant Reviews

## Project Overview
This repository contains an implementation of Aspect-Based Sentiment Analysis (ABSA) for restaurant reviews. The project aims to analyze restaurant reviews at a fine-grained level, extracting sentiments about specific aspects of the restaurant experience (food quality, service, ambiance, etc.) rather than just the overall sentiment of the review.

## Features
- **Multiple Model Implementations**:
  - Transformer-based ABSA models using DistilBERT
  - Baseline sentiment models (Naive Bayes, SVM)
  - Performance comparison between approaches
- **Comprehensive Dataset**:
  - Restaurant reviews with aspect-level annotations
  - Train/Dev/Test splits for robust evaluation
  - Cleaned and preprocessed versions of datasets
- **Evaluation Framework**:
  - Accuracy, F1, and other relevant metrics
  - Performance visualization tools
  - Detailed analysis of model results

## Repository Structure
```
dsa-absa/
├── Dataset/               # Restaurant reviews dataset with ABSA annotations
├── dataset_stats/         # Visualizations of dataset characteristics
├── evaluation/            # Evaluation scripts and metrics
├── logs/                  # Training logs
├── models/                # Model implementations
│   ├── absa/              # Transformer-based ABSA models
│   ├── baseline/          # Naive Bayes baseline models
│   └── svm/               # SVM baseline models
├── results/               # Model checkpoints and evaluation results
├── saved_model/           # Best performing models
├── scripts/               # Utility scripts for visualization and analysis
├── training_plots/        # Training performance visualizations
└── utils/                 # Utility functions for data processing
```

## Dataset
The project uses a restaurant review dataset with aspect-based annotations in the format:
- `tokens`: Tokenized review text
- `absa1`, `absa2`, `absa3`: Aspect annotations including:
  - Position indices of aspect terms
  - Aspect category (e.g., FOOD#QUALITY, SERVICE#GENERAL)
  - Sentiment polarity (0=negative, 1=neutral, 2=positive)

## Model Architecture
The primary ABSA model uses DistilBERT, a lightweight transformer model, fine-tuned for aspect-based sentiment classification. The architecture includes:
- DistilBERT encoder for text representation
- Classification head for sentiment prediction
- Custom data preprocessing for aspect extraction

## Setup and Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/dsa-absa.git
cd dsa-absa

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data and spaCy model
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```

## Usage

### Training the ABSA Model
```bash
cd models/absa
python absa-1.py
```

### Training Baseline Models
```bash
cd models/baseline
python train_baseline.py

# OR for SVM baseline
cd models/svm
python train_svm_baseline.py
```

### Evaluation
```bash
cd evaluation
python evaluate_metrics.py
```

### Visualizing Results
```bash
cd scripts
python plot_training_stats.py
```

### Training all models and Creating Images/Chart
```bash
python evaluation/compare_models.py
```

> **Note:** This command **must be executed from the project root** to ensure proper access to datasets and internal modules.

## Results
The transformer-based ABSA model achieves superior performance compared to baseline models:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| DistilBERT ABSA | ~0.85 | ~0.84 |
| Naive Bayes | ~0.70 | ~0.68 |
| SVM | ~0.75 | ~0.74 |

## License
This project is licensed under the terms of the LICENSE file included in the repository.

## Acknowledgments
- This project was developed as a final project for the Machine Learning course.
- The dataset is based on the SemEval-2016 restaurant reviews dataset.

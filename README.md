# Structural Health Monitoring Using Machine Learning

## Project Overview
This project uses vibration-based modal frequency data to classify the structural condition of a bridge as healthy or damaged.

## Dataset
- Dataset: Yonghe Bridge Modal Frequencies (FDD)
- Input features: 4 modal frequencies
- Total samples: 216
- Healthy samples: 192
- Damaged samples: 24

## Machine Learning Models
The Python script compares multiple classification algorithms:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- XGBoost

## Project Files
- `shm_classifier.py` – Main machine learning implementation
- `Yonghe_Modal_FDD.mat` – Dataset
- `SMC_Modal.m` – MATLAB script
- `requirements.txt` – Required Python libraries

## Installation

Install Python 3, then open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python shm_classifier.py
```

The script generates evaluation results and visualizations.

## Technologies Used
Python, NumPy, Pandas, SciPy, Matplotlib, Seaborn, Scikit-learn and XGBoost.

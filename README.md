# Data Mining Homework 3: Clustering Analysis

Comprehensive clustering analysis on the Wine dataset from the UCI Machine Learning Repository.

## Overview

This project implements multiple clustering algorithms and techniques:
- **K-Means** clustering (from scratch and scikit-learn)
- **Hierarchical Clustering** with stability analysis
- **DBSCAN** with automated parameter estimation
- **Automated Clustering Pipeline** using internal validation metrics

## Files

- `clustering_homework.py` - Main Python implementation with all clustering algorithms
- `report.tex` - LaTeX report with detailed analysis and results
- `report.pdf` - Compiled PDF report
- `eda_*.png` - Exploratory Data Analysis visualizations

## Dataset

- **Wine Dataset** from UCI Machine Learning Repository
- 178 samples, 13 features
- Chemical analysis of wines from different cultivars
- No class labels used (unsupervised learning only)

## Key Features

### Data Preprocessing
- Three dataset versions: Raw, Standardized (Z-score), Min-Max normalized
- Feature statistics and missing value checks

### Exploratory Data Analysis
- Feature histograms and correlation analysis
- PCA visualization (2D projection)
- Highly correlated feature pair analysis

### Algorithms & Analysis

**Part 1: Distance Behavior**
- Euclidean, Manhattan, and Cosine distance metrics
- Distance concentration analysis

**Part 2: K-Means Scaling Impact**
- Combined scoring metric: 0.7 × Silhouette + 0.3 × (1 - Inertia)
- Comparison across different preprocessing methods

**Part 3: K-Means Implementation**
- Custom from-scratch implementation
- Comparison with scikit-learn
- Perfect algorithm alignment verified

**Part 4: Hierarchical Clustering**
- Linkage methods: Single, Complete, Average, Ward
- Cophenetic correlation coefficient for stability ranking

**Part 5: Automated DBSCAN**
- k-nearest neighbor distance analysis
- Epsilon estimation (elbow + statistical methods)
- Feature space modification with PCA

**Part 6: Automated Clustering System**
- Full grid search over algorithms and parameters
- Silhouette score optimization
- Best configuration selection

### Bonus Analyses

**Bonus 1: Subspace Clustering via PCA**
- Evaluate K-Means across PCA dimensions (2-10 components)
- Optimal: 2 components with silhouette score 0.5611

**Bonus 2: Cluster Stability under Noise**
- Adjusted Rand Index (ARI) with Gaussian noise
- σ levels: 0.01, 0.05, 0.10
- Demonstrates excellent cluster robustness (ARI > 0.97)

## Results Summary

**Best Clustering Configuration:**
- Algorithm: Agglomerative Clustering
- Preprocessing: Raw data
- Parameters: k=2, linkage=average
- Silhouette Score: 0.6587

**Reproducibility:**
- Random seed: 61
- All implementations are reproducible
- Complete numerical results in report

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
```

## Usage

```bash
python clustering_homework.py
```

Generates:
- Console output with all numerical results
- EDA visualizations (PNG files)
- Statistics and metrics for each algorithm

## Report

See `report.pdf` for the complete technical report with:
- Methodology explanations
- All results tables
- Analysis and interpretations
- Conclusions and recommendations

## Author

Alireza Jahedi (Student ID: 40435338)

## Date

February 2026

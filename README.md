# 🏠 KC House Price Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Linear%20Regression-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-EDA-150458?logo=pandas&logoColor=white)
![R²](https://img.shields.io/badge/Test%20R%C2%B2-0.90-brightgreen)

Predict home prices in **King County, Washington** (Seattle & surrounding areas) with a machine-learning regression pipeline that reaches a **test R² of 0.90**.

A single end-to-end Jupyter notebook walks through exploratory data analysis, outlier detection, feature engineering, and model evaluation — all on a classic real-world dataset.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Results](#results)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Key Techniques](#key-techniques)

## Features

- **End-to-end EDA** — distributions, boxplots, scatter plots, and correlation heatmaps for every feature
- **Outlier analysis** — an IQR-based helper to quantify outliers per numeric column
- **Feature engineering** — era binning, log transforms, geo-distance features, leakage-safe target encoding, and interaction terms
- **Regression modeling** — linear regression on a log-transformed target with R², MSE, MAE, and MAPE evaluation

## Dataset

[`kc_house_data.csv`](./kc_house_data.csv) contains **21,613** home sale records with **21 columns**:

| Feature | Description |
| --- | --- |
| `price` | Sale price (USD) — target |
| `bedrooms` / `bathrooms` | Room counts |
| `sqft_living` / `sqft_lot` / `sqft_above` / `sqft_basement` | Square footage values |
| `floors`, `waterfront`, `view` | Physical attributes |
| `condition`, `grade` | Quality ratings |
| `yr_built`, `yr_renovated` | Age and renovation year |
| `zipcode`, `lat`, `long` | Location |
| `sqft_living15`, `sqft_lot15` | Neighbor-adjacent footage |

> 💡 This is a well-known Kaggle dataset often used to learn regression pipelines.

## Results

| Metric | Train | Test |
| --- | --- | --- |
| **R²** | 0.882 | **0.900** |
| **MSE** | 15.8B | 13.9B |
| **MAE** | — | $71.7k |
| **MAPE** | — | 13.4% |

## Getting Started

### Prerequisites

- Python 3.9+
- [Jupyter](https://jupyter.org/install)

### Installation

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Running the notebook

```bash
jupyter notebook kc_house_price_prediction.ipynb
```

Select **Kernel → Restart & Run All** to execute the full pipeline, or step through cell-by-cell to follow the analysis.

## How It Works

1. **Load & inspect** the data (`info()`, `describe()`, duplicate & missing-value checks)
2. **Clean** — drop the `id` column and parse the `date` column
3. **Explore** — visualize every feature to spot distributions and relationships
4. **Engineer features** — bin construction eras, encode renovation eras, create log transforms and distance-to-hub geo features, compute per-zipcode aggregates, add interaction and polynomial terms
5. **Filter** — keep only homes with 1–9 bedrooms
6. **Model** — train/test split (80/20), fit linear regression on `log1p(price)`, then invert predictions with `expm1`
7. **Evaluate** — report R², MSE, MAE, and MAPE

## Key Techniques

- **Leakage-safe target encoding** — `zipcode` is mapped to the mean `log(price)` per zipcode, computed **on the training set only**
- **Per-zipcode aggregates** — median living area, average rooms, waterfront ratio, average condition/grade per neighborhood (train-only)
- **Interaction & polynomial terms** — `grade × sqft_living`, `floors × sqft_living`, `home_age²`, `sqft_per_bedroom`, and more, so the linear model captures nonlinear relationships
- **Log-transformed target** — training on `log1p(price)` stabilizes the heavy-tailed price distribution
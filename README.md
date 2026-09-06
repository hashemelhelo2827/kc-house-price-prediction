# 🏠 KC House Price Prediction

Predict home prices in **King County, Washington** (Seattle & surrounding areas) using a machine-learning regression pipeline built in Python.

This project walks through the full data-science workflow inside a single Jupyter notebook: exploratory data analysis, feature engineering, outlier analysis, and a linear regression model with log-target transformation — achieving a **test R² of ~0.90**.

## ✨ Features

- **End-to-end EDA** — distributions, boxplots, scatter plots, and correlation heatmaps for every feature
- **Feature engineering** — built-era / renovation-era binning, basement & living-area log transforms, geo-derived distance-to-hub features, leakage-safe **zipcode target-encoding**, **per-zipcode aggregates** (median sqft, avg bedrooms/bathrooms, waterfront ratio, avg condition/grade), **interaction & polynomial terms** (grade x sqft_living, grade x condition, floors x sqft, sqft_per_bedroom, home_age², etc.), and home-age / renovation features
- **Outlier analysis** — an IQR-based helper to quantify outliers per numeric column
- **Regression modeling** — linear regression on a log-transformed target with R², MSE, MAE, and MAPE evaluation

## 📊 Dataset

[`kc_house_data.csv`](./kc_house_data.csv) contains **21,613** home sale records with **21 columns**, covering:

| Feature | Description |
| --- | --- |
| `price` | Sale price (USD) |
| `bedrooms` / `bathrooms` | Room counts |
| `sqft_living` / `sqft_lot` / `sqft_above` / `sqft_basement` | Square footage values |
| `floors`, `waterfront`, `view` | Physical attributes |
| `condition`, `grade` | Quality ratings |
| `yr_built`, `yr_renovated` | Age and renovation year |
| `zipcode`, `lat`, `long` | Location |
| `sqft_living15`, `sqft_lot15` | Neighbor-adjacent footage |

## 📈 Results

| Metric | Train | Test |
| --- | --- | --- |
| **R²** | 0.882 | **0.900** |
| **MSE** | 15.8B | 13.9B |
| **MAE** | — | ~$71.7k |
| **MAPE** | — | ~13.4% |

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Jupyter](https://jupyter.org/install) (to run notebooks)

### Installation

Install the required packages globally:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Running the notebook

```bash
jupyter notebook kc_house_price_prediction.ipynb
```

Select **Kernel → Restart & Run All** to execute the full pipeline, or step through cell-by-cell to follow the analysis.

> 💡 The notebook assumes `kc_house_data.csv` is in the same directory as the notebook file.

## 🧠 How It Works

1. **Load & inspect** the data (`info()`, `describe()`, duplicate & missing-value checks)
2. **Clean** — drop the `id` column and parse the `date` column
3. **Explore** — visualize every feature to spot distributions and relationships
4. **Engineer features** — bin construction eras, encode renovation eras, create log transforms and distance-to-hub geo features, target-encode zipcode (leakage-safe), compute per-zipcode aggregates, add interaction and polynomial terms
5. **Filter outliers** — keep only homes with 1–9 bedrooms
6. **Model** — train/test split (80/20), fit linear regression on `log1p(price)`, then invert predictions back to the original scale
7. **Evaluate** — report R², MSE, MAE, and MAPE

## 🤝 Contribution

Pull requests are welcome. For significant changes, please open an issue first to discuss what you'd like to improve.

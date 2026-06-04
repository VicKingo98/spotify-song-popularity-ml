# Spotify Song Popularity Predictor — Machine Learning Project

Ensemble ML model to predict song popularity scores (0–100) using Spotify track features. Stacking architecture with LightGBM, XGBoost, CatBoost and HistGradientBoosting, deployed as a serverless REST API on AWS Lambda.

---

## Overview

This project develops an end-to-end machine learning pipeline to predict the popularity score of songs on Spotify, based on audio features, artist statistics, and genre information. The model was built as part of a competitive ML challenge at the Master's in Data Analytics, Universidad de los Andes (2026).

The final solution combines four gradient boosting models in a two-level stacking ensemble with a meta-model XGBoost, achieving an **OOF RMSE of 9.993** — outperforming any individual model by ~0.84 points.

---

## Problem Definition

- **Type:** Supervised regression
- **Target variable:** `popularity` (continuous, range 0–100, mean ≈ 33.26)
- **Dataset:** 79,800 training observations with 20 variables (audio features, artist metadata, genre)
- **Justification:** Popularity is a complex, non-linear phenomenon driven primarily by artist reputation rather than audio characteristics alone, motivating the use of ensemble methods over linear models.

---

## Dataset

The dataset used in this project was provided exclusively for an academic ML competition at Universidad de los Andes and is **not publicly available**. It is therefore not included in this repository.

Key dataset characteristics:
- 79,800 training observations
- Mix of continuous, discrete, boolean and categorical variables
- No missing values
- Target: `popularity` (0–100)

---

## Feature Engineering

~40 features were engineered per observation, including:

- **Artist aggregations** (fit on train fold): mean, median, std, max, p75, range, coefficient of variation, hit rate (% songs with popularity > 60), explicit ratio, genre diversity
- **Genre aggregations** (fit on train fold): mean, median, std, max, count
- **Album aggregations** (fit on train fold): mean and song count
- **Target encoding with Bayesian smoothing** for artist and genre, using Leave-One-Out on train to avoid leakage
- **Text features:** `has_feat`, `has_remix`, `has_live`, `track_name_len`, `n_artists`, `album_is_single`
- **Audio interaction features:** `loudness × energy`, `danceability × valence`, `artist_mean - genre_mean`, `duration_min`, `key_mode`

---

## Methodology

### 1. Preprocessing
- K-Fold Cross-Validation with 5 folds (KFold, shuffle=True, random_state=42)
- All target-based transformations fitted exclusively on training folds and applied to validation and test sets to prevent data leakage

### 2. Base Models & Hyperparameter Tuning
Four models were calibrated using **Optuna** (Bayesian optimization via TPE, 50–75 trials each):

| Model | Individual RMSE (OOF) |
|---|---|
| LightGBM | 10.6285 ± 0.1217 |
| XGBoost | 10.8337 ± 0.0593 |
| CatBoost | 10.6826 ± 0.0712 |
| HistGradientBoosting | 10.6914 ± 0.0845 |

All models used early stopping (50 rounds) to prevent overfitting.

### 3. Stacking Ensemble
A two-level stacking architecture was implemented:

- **Level 1:** Four base models generating Out-Of-Fold (OOF) predictions across all 5 folds
- **Level 2:** Meta-model XGBoost trained on OOF predictions, learning optimal non-linear combination

| Ensemble Variant | RMSE OOF |
|---|---|
| Simple average | 10.5707 |
| Weighted average (RMSE-inverse) | 10.5702 |
| **Meta-model XGBoost (selected)** | **9.9926** |

---

## Results

- **Final RMSE OOF:** 9.993
- **Improvement over best individual model:** ~0.84 points
- **Key finding:** Artist-derived features (`album_mean`, `artists_te`, `artist_max`) are the strongest predictors, outweighing audio characteristics like `danceability` or `tempo` — confirming that song popularity is driven primarily by artist reputation rather than sonic attributes.

---

## Deployment

The model was deployed as a **serverless REST API** on AWS using the following architecture:

```
Trained model → Docker Image → Amazon ECR → AWS Lambda → API Gateway → Public URL
```

**Live API endpoint:**
```
POST https://prh3j6fhq9.execute-api.us-east-1.amazonaws.com/default/predict
```

**Request format (application/json):**
```json
{
  "danceability": 0.516,
  "energy": 0.311,
  "loudness": -13.521,
  "valence": 0.558,
  "speechiness": 0.0306,
  "instrumentalness": 0.000281,
  "liveness": 0.107,
  "acousticness": 0.936,
  "duration_ms": 254306,
  "key": 1,
  "mode": 1,
  "explicit": 0,
  "track_name": "Song Title",
  "artists": "Artist Name",
  "album_name": "Album Name",
  "track_genre": "genre"
}
```

**Response:**
```json
{
  "cancion": "Song Title",
  "artista": "Artist Name",
  "popularidad_predicha": 42.5,
  "interpretacion": "Popularidad media (30-60)"
}
```

> **Note:** The deployed API uses an ensemble of LightGBM and CatBoost (instead of the full 4-model ensemble) due to AWS Lambda memory and package size constraints. Predictions are conservative but functionally correct and publicly accessible.

---

## Project Structure

```
📁 spotify-song-popularity-ml/
│
├── README.md
├── Dockerfile
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── spotify_popularity_predictor.ipynb
│
├── src/
│   ├── features.py
│   ├── lambda_function.py
│   └── test_lambda.py
│
├── modelo/
│   └── (trained model artifacts)
│
└── data/
    └── README.md        ← Dataset access note
```

---

## Requirements

```
lightgbm
xgboost
catboost
scikit-learn
pandas
numpy
optuna
jupyter
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## Limitations & Future Work

- The deployed API uses a reduced 2-model ensemble due to Lambda memory constraints
- Popularity at the extremes (viral hits or unknown songs) is harder to predict, as it depends on external factors not captured in the dataset (social media impact, editorial placement)
- Future improvements: incorporate streaming data, social media features, and extend the model to multi-year datasets

---

## Authors

Developed as part of the Machine Learning and NLP course —
Master's in Data Analytics, Universidad de los Andes (2026).

- Natalia Vindrola
- Victor Osorio
- Javier Cañarte

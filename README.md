# Bishkek Apartment Price Prediction

**Predicting apartment prices in Bishkek** using feature engineering and multiple machine learning models (CatBoost, Random Forest, Decision Tree, Linear Regression).  
This repository contains notebooks, scripts, and trained model artifacts used to preprocess data, train models, evaluate them, and produce submissions.

---

## 🔎 Project Overview

The goal is to predict apartment prices (target column: `usd_price`) for listings in Bishkek. The dataset contains listing text, location, building and apartment attributes, and marketplace metadata. We focus on careful preprocessing and feature engineering to make models robust on limited data.

---

## 🧰 What I did (high-level)

1. **Drop low-value / sparse columns** (columns with many missing values or low predictive power), for example:
   - `Питьевая вода`, `Электричество`, `Канализация`, `Площадь участка`, `Возможность рассрочки`, `Возможность обмена`, `Пол`, `Телефон`, `Интернет`, `Возможность ипотеки`, `Парковка`, `Безопасность`, `Мебель`, `Балкон`, `Разное`, `Газ`, `Входная дверь`, `Санузел`, `Правоустанавливающие документы`, `Высота потолков`

2. **Feature extraction / engineering**
   - `num_room` — extracted from Russian descriptions (`main`) covering studios, "6 и более", free layouts, etc.
   - `area` — parsed from `Площадь` strings (first `м2` found).
   - `days_since_added`, `hours_since_lifted` — parsed from `added` and `upped`.
   - `Материал_дома`, `Год_постройки` — parsed from `Дом`.
   - `Текущий_этаж`, `Всего_этажей`, `Относительный_этаж` — parsed from `Этаж`.
   - Impute numeric missing values with column means (documented and optional to change to median or robust methods).
   - Fill categorical missing values with `"NotGiven"`.

3. **Modeling**
   - Baselines: Linear Regression, Ridge/Lasso (pipelines with ColumnTransformer).
   - Tree-based: Decision Tree, Random Forest.
   - Gradient-tree boosting: CatBoost (uses native categorical support), with Optuna tuning for MAPE/RMSE.
   - Optional: HDBSCAN used as a clustering feature (cluster label appended as categorical).

4. **Evaluation**
   - R², MAE, MAPE, MSE/RMSE
   - Cross-validation used to assess performance and avoid leakage
   - Final validation on held-out test split and submission generation

---


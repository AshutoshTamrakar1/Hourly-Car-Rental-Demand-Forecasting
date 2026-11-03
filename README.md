# Hourly Car Rental Demand Forecasting (LightGBM & Time-Series Feature Engineering)

> **End-to-end time-series demand forecasting pipeline for hourly car rentals in Bangalore (Analytics Vidhya Job-A-Thon April 2022 — Top 70 Private Leaderboard), featuring cyclical Fourier encodings, leakage-free target aggregations, `TimeSeriesSplit` cross-validation, and LightGBM objective blending.**

---

## Problem Statement
ABC is a car rental company based out of Bangalore. It rents cars for both in and out stations at affordable prices. The users can rent different types of cars like Sedans, Hatchbacks, SUVs and MUVs, Minivans and so on.

In recent times, the demand for cars is on the rise. As a result, the company would like to tackle the problem of supply and demand. The ultimate goal of the company is to strike the balance between the supply and demand inorder to meet the user expectations. The company has collected the details of each rental. Based on the past data, the company would like to forecast the demand of car rentals on an hourly basis.

## Objective
The main objective of the problem is to develop the machine learning approach to forecast the demand of car rentals on an hourly basis.

## Approach
The data given had no features except date and hour. This was clearly a time series problem, so first thing I did is to reduce this problem to a regression problem. In order to reduce it to a Regression, I tried to capture the temporal dynamics of the data like things that affect every object in the data. So I extracted time based features like day of week, week of month, week of year, is the day is weekend, or is it start of the month or end of month because maybe people prefer to go out around payday or on public holidays.
After extracting all these features, I could approach it like simple Machine Learning problem.

## ML Approach
Before building an ML model, feature engineering was done. Basic time related features such as day, month, hour, weekofyear, dayofyear, weekday were created.I have tried some tree based model for this problem like Catboost, XGboost and LightGBM, out of these, LightGBM was working best, so I did some parameter tuning and checked the results for best parameter combination.The evaluation metric was 'RMSE'. This model gave a public leaderboard score of 33.88 and a private leaderboard score of 33.928.

## Final Result
Private Leaderboard Rank - 70

Public Leaderboard Rank - 127

---

## Refactored Pipeline & Improved Results

### Key Improvements Made
1. **Corrected & Expanded Feature Engineering (8 → 29 Features):**
   - **Retained Crucial Temporal Signals:** Kept `wday` (day of week) and `year` in the training features (which were previously dropped prior to model fitting) and added a continuous trend feature (`days_since_start`) plus `dayofyear`.
   - **Fixed Weekend Logic Bug:** Corrected `is_wknd` from `weekday // 4` (which misclassified Friday as a weekend) to `weekday >= 5` (Saturday & Sunday) and added a dedicated `is_friday` flag, `is_payday_window`, `part_of_day`, and `is_peak_hour`.
   - **Cyclical Fourier Encodings:** Added `sin` and `cos` transformations for `hour`, `wday`, `month`, and `dayofyear` to preserve periodic continuity across boundaries (e.g., `hour=23` to `hour=0`).
2. **Leakage-Free Target Aggregations:**
   - Computed out-of-sample historical demand statistics (`mean`, `std`, `median` grouped by `(hour, wday)` and `(hour, month)`) strictly on the training split before mapping onto validation and test sets.
3. **Validation Strategy & Full-Dataset Retraining:**
   - Evaluated both `regression` (L2) and `poisson` LightGBM objectives alongside a 50/50 ensemble blend on the chronological 28-day holdout split (`17,602` train / `645` validation rows) and across 5-fold `TimeSeriesSplit` cross-validation.
   - Retrained final models on **100% of the training data (`18,247` rows)** so the final submission incorporates the most recent 28 days of demand immediately preceding the test period (`2021-03-01` onwards).

### Validation Performance Comparison (28-Day Holdout Split)

| Model | Features | Best Iteration | Validation RMSE | Improvement vs. Original |
| :--- | :---: | :---: | :---: | :---: |
| Naive Baseline (`Hour × Weekday` Historical Mean) | 2 | — | `31.3326` | `+0.1986` |
| **Original LightGBM (`poisson`)** | 8 | 73 | **`31.1340`** | — |
| **Refactored LightGBM (`regression` L2)** | 29 | 25 | **`30.4374`** | **`-0.6966`** |
| **Refactored LightGBM (`50/50 Blend` L2 + Poisson)** | 29 | — | **`30.3993`** | **`-0.7347`** |
| **Refactored LightGBM (`poisson`)** | 29 | 55 | **`30.3894`** | **`-0.7446` (~2.4% reduction)** |

### 5-Fold `TimeSeriesSplit` Cross-Validation
Across 5 sequential 28-day (`645`-row) validation windows, the refactored model achieved:
- **Fold 1:** `40.1678` | **Fold 2:** `33.4466` | **Fold 3:** `37.8521` | **Fold 4:** `33.8853` | **Fold 5:** `30.4374`
- **Mean 5-Fold RMSE:** `35.1579 ± 3.4406`

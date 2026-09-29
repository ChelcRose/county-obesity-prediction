# Final Results Summary for Manuscript

**Status:** Approved final analysis  
**Finalized:** 29 September 2026  
**Primary outcome:** County-level age-adjusted adult obesity prevalence (`OBESITY_AdjPrev`), expressed in percentage points

This document is the consolidated reference for the final paper. The approved early-stopping training procedure is treated simply as the final XGBoost procedure; manuscript-facing model names should be **Linear Regression**, **Random Forest**, **XGBoost**, and **Stacking Ensemble**, without Version 1 or Version 2 labels.

## 1. Study sample and predictors

- The integrated modeling sample contains **3,135 counties**.
- The initial candidate set contains **18 USDA Food Environment Atlas predictors**.
- The target is the CDC PLACES age-adjusted adult obesity prevalence estimate, `OBESITY_AdjPrev`.
- County records are joined by five-character FIPS.
- USDA `-8888` records were handled by the established county-exclusion rule; `-9999` values were treated as missing.
- Predictors with more than 20% missingness in the applicable training data were excluded.
- Remaining missing values were median-imputed using the applicable training data.
- Pearson correlation screening used an absolute threshold of 0.80 and the predefined pair-resolution rules.
- IQR values were calculated for review only; observations were not removed, winsorized, or transformed based on IQR.
- Linear Regression inputs were standardized using training-fitted parameters. Tree-model inputs were not standardized.

The final **14 retained predictors** were:

1. `PCT_LACCESS_LOWI19`
2. `CONVSPTH20`
3. `FFRPTH20`
4. `FSRPTH20`
5. `MEDHHINC21`
6. `POVRATE21`
7. `DEEPPOVRATE21`
8. `PC_SNAPBEN22`
9. `PCT_65OLDER20`
10. `PCT_18YOUNGER20`
11. `PCT_NHWHITE20`
12. `PCT_NHBLACK20`
13. `PCT_HISP20`
14. `PCT_NHASIAN20`

The two variables excluded for greater than 20% missingness were `GROCPTH20` and `RECFACPTH20`. The predefined correlation decisions excluded `CHILDPOVRATE21` in favor of `POVRATE21` and `PCT_LACCESS_POP19` in favor of `PCT_LACCESS_LOWI19`.

## 2. Final validation and modeling procedure

- Five-fold shuffled outer cross-validation evaluated generalization.
- Each outer fold contained 2,508 training counties and 627 untouched validation counties.
- Three-fold shuffled inner cross-validation tuned Random Forest and XGBoost.
- All splitters used `random_state=42`.
- Hyperparameter selection optimized `neg_root_mean_squared_error`.
- Randomized searches used 10 sampled configurations.
- Outer-fold MAE, RMSE, and R² were calculated, then summarized using the arithmetic mean and sample SD across five folds.
- The folds were random county-level folds, not spatially blocked, state-held-out, temporal, or repeated cross-validation folds.

### Models

**Linear Regression:** Ordinary least squares `LinearRegression`, with median imputation, the predefined screening rules, and standardization. No hyperparameters were tuned.

**Random Forest:** `RandomForestRegressor` with the established preprocessing. The randomized search evaluated:

- `n_estimators`: 100, 200, 300, 400, 500
- `max_depth`: None, 5, 10, 15, 20, 30
- `min_samples_split`: 2, 5, 10

**XGBoost:** `XGBRegressor(objective="reg:squarederror")` with the established preprocessing and leakage-safe early stopping. The randomized search evaluated:

- `learning_rate`: 0.01, 0.05, 0.10, 0.20
- `max_depth`: 2, 3, 4, 5, 6
- `subsample`: 0.6, 0.8, 1.0

`n_estimators` was not included in the randomized search. Each XGBoost fit used:

- `eval_metric="rmse"`
- `early_stopping_rounds=50`
- Internal early-stopping validation fraction of 0.20
- Maximum boosting ceiling of 2,000 trees
- `random_state=42`

The early-stopping evaluation subset was created only from the training data available to that specific fit. Inner validation folds, stacking OOF validation folds, and outer validation folds were never used as early-stopping evaluation data. After identifying `best_iteration`, preprocessing and XGBoost were refitted on all permissible training rows with `n_estimators = best_iteration + 1`.

Across the five final outer-training refits, XGBoost selected `learning_rate=0.01`, `max_depth=6`, and `subsample=0.6`. Effective tree counts were 628, 553, 581, 554, and 809. No fit reached the 2,000-tree ceiling.

### Fold-specific selected settings and XGBoost diagnostics

| Outer fold | RF trees | RF max depth | RF min split | XGB learning rate | XGB depth | XGB subsample | Best inner-CV RMSE | XGB best iteration | Effective trees | Early-stopping validation RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | None | 2 | 0.01 | 6 | 0.6 | 2.7649 | 627 | 628 | 2.8630 |
| 2 | 100 | 20 | 5 | 0.01 | 6 | 0.6 | 2.8254 | 552 | 553 | 2.8387 |
| 3 | 100 | 20 | 5 | 0.01 | 6 | 0.6 | 2.8237 | 580 | 581 | 2.9436 |
| 4 | 100 | 20 | 5 | 0.01 | 6 | 0.6 | 2.8863 | 553 | 554 | 2.9005 |
| 5 | 100 | None | 2 | 0.01 | 6 | 0.6 | 2.8496 | 808 | 809 | 2.8197 |

The early-stopping validation RMSE values above belong to the internal 20% subsets used during each final outer-training refit. They are training diagnostics, not outer-fold performance estimates.

**Stacking Ensemble:** Linear Regression, Random Forest, and XGBoost were the base learners; ordinary least squares Linear Regression was the meta-learner. Inside each outer-training set, a separate shuffled three-fold procedure generated OOF base predictions for meta-learner training. Every county's meta-training prediction came from base models that were not fitted on that county. XGBoost additionally created its early-stopping subset only within each stacking-training portion and was then refitted on all rows in that stacking-training portion. The fitted base learners were refitted on the complete outer-training set before predicting the untouched outer validation fold.

RF and XGBoost hyperparameters were selected using the complete outer-training set before the separate stacking OOF fits. Consequently, a stacking-OOF row's target could influence the shared hyperparameter choice even though the row was excluded from its base-model fit and from the XGBoost early-stopping subset. This does not contaminate the untouched outer validation fold, but the manuscript should not claim that hyperparameters were reselected independently inside every stacking-training split.

## 3. Final aggregate performance

| Model | MAE, mean ± SD | RMSE, mean ± SD | R², mean ± SD |
|---|---:|---:|---:|
| Linear Regression | 2.4105 ± 0.0781 | 3.0956 ± 0.1117 | 0.5491 ± 0.0407 |
| Random Forest | 2.2231 ± 0.1135 | 2.8832 ± 0.1648 | 0.6098 ± 0.0297 |
| XGBoost | 2.1708 ± 0.0940 | 2.7945 ± 0.1415 | 0.6333 ± 0.0266 |
| **Stacking Ensemble** | **2.1681 ± 0.0871** | **2.7830 ± 0.1353** | **0.6362 ± 0.0281** |

The Stacking Ensemble had the lowest mean MAE and RMSE and the highest mean R². These are descriptive comparisons; no statistical significance test was performed.

## 4. Final outer-fold performance

| Fold | Model | MAE | RMSE | R² |
|---:|---|---:|---:|---:|
| 1 | Linear Regression | 2.4749 | 3.1798 | 0.5456 |
| 1 | Random Forest | 2.2954 | 2.9629 | 0.6055 |
| 1 | XGBoost | 2.2656 | 2.9068 | 0.6203 |
| 1 | Stacking Ensemble | **2.2615** | **2.8932** | **0.6239** |
| 2 | Linear Regression | 2.4559 | 3.0924 | 0.5906 |
| 2 | Random Forest | 2.3576 | 3.0443 | 0.6032 |
| 2 | XGBoost | 2.2505 | 2.8775 | 0.6455 |
| 2 | Stacking Ensemble | **2.2288** | **2.8473** | **0.6529** |
| 3 | Linear Regression | 2.4488 | 3.2093 | 0.5307 |
| 3 | Random Forest | 2.2401 | 2.9819 | 0.5948 |
| 3 | XGBoost | **2.1909** | 2.8979 | 0.6173 |
| 3 | Stacking Ensemble | 2.1936 | **2.8878** | **0.6200** |
| 4 | Linear Regression | 2.2829 | 2.9242 | 0.5859 |
| 4 | Random Forest | 2.0736 | 2.6468 | 0.6608 |
| 4 | XGBoost | 2.0554 | 2.5938 | 0.6742 |
| 4 | Stacking Ensemble | **2.0538** | **2.5842** | **0.6766** |
| 5 | Linear Regression | 2.3902 | 3.0724 | 0.4927 |
| 5 | Random Forest | 2.1486 | 2.7801 | 0.5846 |
| 5 | XGBoost | **2.0913** | **2.6964** | **0.6093** |
| 5 | Stacking Ensemble | 2.1030 | 2.7024 | 0.6076 |

Stacking had the best RMSE and R² in four of five folds and the best MAE in three of five folds. XGBoost had the best MAE in folds 3 and 5 and the best RMSE and R² in fold 5.

## 5. Descriptive improvement of stacking

Relative to Linear Regression, the Stacking Ensemble reduced mean MAE by 0.2424 (10.0560%) and mean RMSE by 0.3126 (10.0998%), while mean R² was higher by 0.0871.

Relative to Random Forest, the Stacking Ensemble reduced mean MAE by 0.0549 (2.4702%) and mean RMSE by 0.1002 (3.4767%), while mean R² was higher by 0.0264.

Relative to XGBoost, the Stacking Ensemble reduced mean MAE by 0.0026 (0.1203%) and mean RMSE by 0.0115 (0.4114%), while mean R² was higher by 0.0029. The advantage over XGBoost was small and must not be characterized as statistically significant.

## 6. Final feature importance

Feature importance came from separate descriptive models fitted to all 3,135 counties; these models were not used to estimate held-out performance. Random Forest importance is mean decrease in impurity. XGBoost importance is normalized gain. Importance does not establish direction, statistical association, or causality.

The descriptive XGBoost model used the final structural settings (`learning_rate=0.01`, `max_depth=6`, `subsample=0.6`). An internal 80/20 split selected best iteration 524 with early-stopping validation RMSE 2.9028. Preprocessing and XGBoost were then refitted on all counties using 525 trees.

### Random Forest top five — mean decrease in impurity

| Rank | Predictor | Importance |
|---:|---|---:|
| 1 | `MEDHHINC21` | 0.3448 |
| 2 | `PCT_18YOUNGER20` | 0.1062 |
| 3 | `PCT_NHBLACK20` | 0.0927 |
| 4 | `FSRPTH20` | 0.0653 |
| 5 | `PCT_NHASIAN20` | 0.0623 |

### XGBoost top five — normalized gain

| Rank | Predictor | Importance |
|---:|---|---:|
| 1 | `MEDHHINC21` | 0.3294 |
| 2 | `PCT_18YOUNGER20` | 0.1014 |
| 3 | `POVRATE21` | 0.0860 |
| 4 | `PCT_NHBLACK20` | 0.0837 |
| 5 | `PCT_NHASIAN20` | 0.0696 |

Four predictors appeared in both top-five lists: `MEDHHINC21`, `PCT_18YOUNGER20`, `PCT_NHBLACK20`, and `PCT_NHASIAN20`.

## 7. Final county mapping results

- The final mapping source contains **3,135 held-out Stacking Ensemble predictions**.
- There is exactly one prediction for each of 3,135 unique county FIPS.
- All 3,135 prediction FIPS matched Census county geometry.
- There were no missing geometries, predictions, targets, or prediction differences.
- The boundary source contained 3,235 geometries; 100 boundaries had no result because they were outside the modeling sample.
- Geometry remained in CRS EPSG:4269.
- `Prediction_Difference` equals predicted minus observed obesity prevalence.
- The final prediction-difference range was **−9.4675 to +11.3094 percentage points**.
- These map values are combined held-out predictions from five fold-specific ensemble models, not predictions from one production model fitted on the complete dataset.

## 8. Recommended manuscript interpretation

The final analysis found that the Stacking Ensemble had the best mean predictive performance under the selected random county-level five-fold cross-validation design, with MAE 2.1681 ± 0.0871, RMSE 2.7830 ± 0.1353, and R² 0.6362 ± 0.0281. XGBoost performed almost identically, and the ensemble's numerical advantage over XGBoost was small. The results support describing stacking as the best-performing model in this experiment, but they do not establish a statistically significant improvement.

Feature importance indicates which predictors the fitted tree models used most strongly for prediction. It does not establish that these variables cause obesity prevalence, nor does it provide the direction of their relationship with the outcome.

The reported uncertainty is the sample SD across five folds of one shuffled partition. It is not a confidence interval, repeated-CV uncertainty estimate, or hypothesis test. Random county folds also do not directly estimate performance for wholly unseen states, spatial regions, or future time periods.

The analysis is ecological and combines county predictors from different reference years with the CDC target. Results should therefore be described as county-level predictive associations, not individual-level relationships or contemporaneous causal effects.

## 9. Manuscript-ready results paragraph

> The final analytic sample included 3,135 counties and 14 retained predictors. Under five-fold outer cross-validation, the Stacking Ensemble achieved the best mean predictive performance, with an MAE of 2.1681 ± 0.0871 percentage points, RMSE of 2.7830 ± 0.1353 percentage points, and R² of 0.6362 ± 0.0281. XGBoost performed similarly (MAE 2.1708 ± 0.0940, RMSE 2.7945 ± 0.1415, R² 0.6333 ± 0.0266), followed by Random Forest (MAE 2.2231 ± 0.1135, RMSE 2.8832 ± 0.1648, R² 0.6098 ± 0.0297) and Linear Regression (MAE 2.4105 ± 0.0781, RMSE 3.0956 ± 0.1117, R² 0.5491 ± 0.0407). The ensemble's advantage over XGBoost was small and was not tested for statistical significance. In descriptive full-data feature-importance models, median household income ranked first for both Random Forest mean-decrease-in-impurity importance and XGBoost gain importance. These rankings describe model usage and should not be interpreted causally.

## 10. Canonical final files

### Performance

- `outputs/model_comparison_final/final_model_performance_comparison.csv`
- `outputs/model_comparison_final/final_all_model_outer_fold_metrics.csv`
- `outputs/model_comparison_final/final_stacking_improvement.csv`
- `outputs/model_comparison_final/final_best_model_by_fold.csv`
- `outputs/model_comparison_final/final_metric_winners.csv`

### Feature importance

- `outputs/feature_importance_final/final_rf_feature_importance.csv`
- `outputs/feature_importance_final/final_rf_top5_features.csv`
- `outputs/feature_importance_final/final_xgb_gain_feature_importance.csv`
- `outputs/feature_importance_final/final_xgb_top5_features.csv`
- `outputs/feature_importance_final/final_feature_importance_metadata.csv`

### Mapping and application data

- `outputs/map_final/final_county_map_data.csv`
- `outputs/map_final/final_county_map_data.geojson`
- `outputs/map_final/final_mapping_verification.csv`
- `outputs/map_final/prediction_fips_missing_in_boundaries.csv`

### Training provenance

- `outputs/xgboost_v2/xgb_v2_outer_fold_results.csv`
- `outputs/xgboost_v2/xgb_v2_outer_predictions.csv`
- `outputs/stacking_v2/stacking_v2_outer_fold_metrics.csv`
- `outputs/stacking_v2/stacking_v2_outer_predictions.csv`
- `outputs/stacking_v2/stacking_v2_best_hyperparameters.csv`
- `outputs/stacking_v2/stacking_v2_xgb_oof_diagnostics.csv`
- `outputs/stacking_v2/stacking_v2_leakage_checks.csv`

The older output directories remain preserved for audit history, but the files listed above are the authoritative final-paper and application inputs.

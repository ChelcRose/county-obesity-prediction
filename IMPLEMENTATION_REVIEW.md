# County obesity prediction: verified implementation review

This report is based on the ten numbered notebooks, their saved outputs, the interim modeling table, the raw data files, and the local XGBoost implementation inspected on 21 September 2026. It describes executed code and saved artifacts; it does not treat notebook comments as evidence that an action occurred. Numerical results below are from saved CSVs unless a calculation is explicitly labeled as a verification calculation. The only repository file added for this request is this report.

## 1. Study / implementation overview

The project predicts county adult obesity prevalence (`OBESITY_AdjPrev`, percentage points) from county food environment and demographic or socioeconomic variables. The target comes from the CDC PLACES 2024 release county file. Eighteen candidate predictors come from the USDA Food Environment Atlas. The pipeline selects those variables from a long USDA table, joins them by five-character FIPS to the CDC target, applies training-fold preprocessing, compares linear regression (LR), random forest (RF), XGBoost (XGB), and a three-model stacking ensemble, then produces descriptive feature importance and county geometry files. The integrated modeling sample is **3,135 counties, 18 candidate predictors, one target**. Each evaluated fold retains **14 predictors** after the implemented screens. The final output is five-fold out-of-fold (OOF) predictions for all 3,135 counties, not predictions from a single model fitted on all counties.

The workflow is in `notebooks/01_data_inspection.ipynb` through `notebooks/10_county_map_preparation.ipynb`. Notebook 03 demonstrates fold preprocessing; notebooks 04–07 implement the model evaluations separately; notebook 08 compares their saved metrics; notebook 09 fits full-data descriptive importance models; notebook 10 joins OOF stacking predictions to boundaries.

## 2. Data sources and integration

| Source | Project file | Variables actually used |
|---|---|---|
| USDA Food Environment Atlas, download documented as July 2025 | `data/raw/StateAndCountyData.csv` | `FIPS`, `State`, `County`, `Variable_Code`, `Value`; the 18 codes below |
| USDA variable metadata | `data/raw/VariableList.csv` and `data/raw/ReadMeFile2025.txt` | Names, categories and units for inspection, not model features |
| CDC PLACES county data, 2024 release | `data/raw/PLACES__County_Data_(GIS_Friendly_Format),_2024_release_20260824.csv` | `CountyFIPS`, `StateAbbr`, `CountyName`, `OBESITY_AdjPrev`; only FIPS and target enter the final merge |
| Census county boundaries | `data/boundaries/cb_2024_us_county_500k/cb_2024_us_county_500k.shp` and associated files | `GEOID`, names, state labels, geometry for mapping |

The 18 initial USDA predictors were `PCT_LACCESS_POP19`, `PCT_LACCESS_LOWI19`, `GROCPTH20`, `CONVSPTH20`, `FFRPTH20`, `FSRPTH20`, `MEDHHINC21`, `POVRATE21`, `CHILDPOVRATE21`, `DEEPPOVRATE21`, `PC_SNAPBEN22`, `PCT_65OLDER20`, `PCT_18YOUNGER20`, `PCT_NHWHITE20`, `PCT_NHBLACK20`, `PCT_HISP20`, `PCT_NHASIAN20`, and `RECFACPTH20`. Their years vary from 2019 to 2022; the notebook did not harmonize measurement years.

The USDA source has 957,753 long-format rows, 3,156 non-null unique FIPS and 304 variable codes. Its 475 rows with null FIPS concern five Alaska labels and contain no selected predictor rows. The CDC file has 3,144 unique FIPS and no missing target. The preparation notebook selected 56,685 USDA rows and converted USDA numeric FIPS through nullable integer to strings padded to five characters; CDC `CountyFIPS` was also string-padded. It excluded all 13 FIPS with at least one selected USDA `-8888` value, then pivoted to 3,143 USDA FIPS with 18 columns. An inner one-to-one FIPS merge with CDC yielded **3,135** records, with no missing target or duplicate FIPS. State and county names matched exactly for all merged rows in the notebook check.

The eight USDA-only FIPS excluded by the inner join are the older Connecticut county codes `09001`, `09003`, `09005`, `09007`, `09009`, `09011`, `09013`, `09015`. The nine CDC-only FIPS are Connecticut planning-region codes `09110`, `09120`, `09130`, `09140`, `09150`, `09160`, `09170`, `09180`, `09190`. These were not crosswalked or aggregated. The other four `-8888` FIPS excluded before this comparison are `02261`, `02270`, `46113`, and `51515`; the nine Connecticut planning-region codes account for the remainder of the 13. Thus the 13 pre-merge FIPS exclusions and the 8+9 unmatched sets should not be added together as if they were disjoint final-sample losses.

## 3. Data cleaning and preprocessing

**Preparation order.** The code selected the 18 USDA codes, formatted FIPS, identified every FIPS containing `-8888` in a selected code and removed that entire FIPS from the selected USDA table (111 flagged cells across 13 FIPS), replaced `-9999` with `NaN` (3,855 cells), pivoted, and inner-joined to CDC. The interim `data/interim/modeling_data.csv` has 3,135 rows, 22 columns (FIPS, state, county, 18 predictors, target). No predictor imputation was performed when this table was saved.

**Missingness screen.** In each training subset, a feature was excluded when its missing percentage was **strictly greater than 20%**; percentages were rounded to two decimals before the comparison in the model preprocessors. Every outer fold excluded `GROCPTH20` and `RECFACPTH20`. Their pre-merge wide-table missingness was 909/3,143 (**28.92%**) and 1,849/3,143 (**58.83%**), respectively. In the final saved 3,135-county table, verification gives **28.995%** and **58.979%**. The other conspicuous pre-merge missingness values were `FFRPTH20` 14.99%, `CONVSPTH20` 9.04%, `FSRPTH20` 8.97%, and `PC_SNAPBEN22` 1.69%. The screen was recalculated during every pipeline fit, including inner search and stacking OOF fits.

**Imputation and correlation.** A `SimpleImputer(strategy="median")` was fitted on the current training subset after missingness screening, then applied to held-out rows. Pearson correlations of the imputed training predictors were screened at absolute **r ≥ 0.80**. The implementation only removes members of two predefined pairs if that pair crosses the threshold: `CHILDPOVRATE21` is removed while `POVRATE21` is retained; `PCT_LACCESS_POP19` is removed while `PCT_LACCESS_LOWI19` is retained. The code does not perform a generic automatic elimination of every highly correlated pair or document an empirical selection rule for which member to keep. A read-only verification from the saved table reproduces these as the only threshold-crossing pairs in all five outer training sets: low-access pair r = **0.8824, 0.8809, 0.8814, 0.8764, 0.8741** by fold; poverty pair r = **0.9368, 0.9388, 0.9392, 0.9388, 0.9369**. The exact displayed values are rounded from that verification, not saved correlation artifacts.

**IQR and scaling.** For each retained training predictor the notebooks calculate Q1, Q3, IQR and the 1.5-IQR fences and count out-of-fence values. They do **not** drop, winsorize, or transform these observations; `implausible_values_removed` is zero in all outer folds. The demonstration notebook reports 1,981, 1,957, 1,998, 1,988 and 1,971 flagged feature values across the five outer training folds, with all 14 features flagging at least one value. These are feature-value counts, not distinct county counts. LR standardizes the imputed, retained features with `StandardScaler` fitted on the relevant training data. RF and XGB use the imputed values without scaling. In the stand-alone LR notebook, scaling is fit on the outer training fold. In the stacking LR pipeline, it is refit inside each OOF training fold. No target-derived feature selection, imputation or scaling is fitted on outer validation data.

The 14 predictors retained in each outer fold, and in notebook 09's full-data importance fit, are `PCT_LACCESS_LOWI19`, `CONVSPTH20`, `FFRPTH20`, `FSRPTH20`, `MEDHHINC21`, `POVRATE21`, `DEEPPOVRATE21`, `PC_SNAPBEN22`, `PCT_65OLDER20`, `PCT_18YOUNGER20`, `PCT_NHWHITE20`, `PCT_NHBLACK20`, `PCT_HISP20`, `PCT_NHASIAN20`.

## 4. Cross-validation and experimental design

All evaluated models use shuffled `KFold(n_splits=5, random_state=42)` on the same ordered modeling table. Each outer split has **2,508 training and 627 validation counties**; each county is validated once. RF and XGB run `RandomizedSearchCV` with shuffled three-fold inner `KFold(..., random_state=42)` and negative RMSE scoring. In each search candidate's inner split, a pipeline fits its missingness screen, median imputer and correlation decisions only on that inner training split. `RandomizedSearchCV` refits the selected pipeline on the complete outer training fold before the outer validation prediction. LR has no hyperparameter search; its preprocessing and fit occur on the outer training fold. The stacking procedure adds a separate three-fold OOF split inside each outer training set (section 6). Outer validation targets and predictors are not used to fit those estimators. Consequently, the reported metrics are out-of-sample with respect to each county's outer fold, conditional on this single random five-fold partition. The study does not use spatially blocked or state-held-out validation.

## 5. Individual models

| Model | Pipeline and configuration | Search |
|---|---|---|
| LR | Training-fitted missingness screen → median imputer → pairwise correlation decisions → IQR count only → `StandardScaler` → default `LinearRegression()` | No tuned hyperparameters, no inner search |
| RF | `ThesisPreprocessor` → `RandomForestRegressor(random_state=42, n_jobs=-1)` | `RandomizedSearchCV`, 10 draws, three inner folds, negative RMSE; `n_estimators`: 100/200/300/400/500; `max_depth`: None/5/10/15/20/30; `min_samples_split`: 2/5/10 |
| XGB | `ThesisPreprocessor` → `XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)` | `RandomizedSearchCV`, 10 draws, three inner folds, negative RMSE; `learning_rate`: .01/.05/.1/.2; `n_estimators`: 100/200/300/400/500; `max_depth`: 2/3/4/5/6; `subsample`: .6/.8/1.0 |

Each search uses `random_state=42` and `n_jobs=-1`. Other estimator parameters are library defaults. The saved best parameters are:

| Outer fold | RF (`n_estimators`, `max_depth`, `min_samples_split`) | XGB (`learning_rate`, `n_estimators`, `max_depth`, `subsample`) |
|---:|---|---|
| 1 | 100, None, 2 | .05, 200, 4, .8 |
| 2 | 100, 20, 5 | .05, 200, 4, .8 |
| 3 | 100, 20, 5 | .05, 200, 4, .8 |
| 4 | 100, 20, 5 | .05, 200, 4, .8 |
| 5 | 100, None, 2 | .05, 200, 4, .8 |

These values occur in both individual-model hyperparameter files and `stacking_best_hyperparameters.csv`. The corresponding individual-model best inner-CV RMSEs are RF **2.8515, 2.9176, 2.9012, 2.9623, 2.9410** and XGB **2.7825, 2.8382, 2.8264, 2.8995, 2.8896** by outer fold. They are selection scores, not the held-out outer results.

## 6. Stacking ensemble

The three base learners are the LR, RF and XGB pipelines above; the meta-learner is default unregularized `LinearRegression()` applied to three base prediction columns. For **each outer fold**, RF and XGB parameters were first selected by separate 10-draw, three-fold inner searches on that fold's **outer training set**, using negative RMSE. A second shuffled three-fold `KFold(random_state=42)` then split the outer training set for stacking OOF predictions. On each stacking split, each base pipeline was cloned and fitted only on the stacking-training portion, with the RF/XGB parameters chosen from the preceding searches. Its predictions for the stacking-validation portion populated `LR_pred`, `RF_pred`, and `XGB_pred`. These 2,508 three-column OOF predictions trained the linear meta-learner against the corresponding outer-training targets.

The base pipelines were subsequently cloned and refitted on the entire 2,508-row outer training set, predicted the 627 untouched outer validation rows, and supplied those three predictions to the trained meta-learner. The resulting stacking predictions supplied outer-fold metrics and one county record each. Thus **inner CV selects RF/XGB parameters; stacking OOF predictions train the meta-learner; outer CV evaluates the complete procedure**. Every preprocessing fit in a base pipeline is limited to the training rows of that fit.

A methodological nuance should be stated precisely: the chosen RF/XGB hyperparameters were selected using the entire outer training set before the stacking OOF base fits, so a stacking-OOF row's target may have influenced hyperparameter choice even though it did not enter that row's base-model fit. This is within the outer training fold and does not contaminate outer validation metrics, but the meta-training predictions are not independent of parameter selection. The code does not re-tune RF/XGB separately within each stacking-training split. The five saved meta coefficients and intercepts are in `outputs/stacking/stacking_meta_coefficients.csv`; these are fold-specific descriptions, not a deployed single meta-model.

## 7. Model evaluation results

Saved `outputs/model_comparison/model_performance_comparison.csv` contains arithmetic means and **sample SD** (`pandas.Series.std()`) over the five outer-fold metric values. MAE and RMSE are in obesity percentage points; R² is unitless.

| Model | MAE mean ± SD | RMSE mean ± SD | R² mean ± SD |
|---|---:|---:|---:|
| Linear regression | 2.4105 ± 0.0781 | 3.0956 ± 0.1117 | 0.5491 ± 0.0407 |
| Random forest | 2.2231 ± 0.1135 | 2.8832 ± 0.1648 | 0.6098 ± 0.0297 |
| XGBoost | 2.1971 ± 0.0958 | 2.8193 ± 0.1497 | 0.6266 ± 0.0311 |
| Stacking ensemble | **2.1817 ± 0.0916** | **2.8027 ± 0.1438** | **0.6309 ± 0.0315** |

| Fold | LR MAE / RMSE / R² | RF MAE / RMSE / R² | XGB MAE / RMSE / R² | Stack MAE / RMSE / R² |
|---:|---|---|---|---|
| 1 | 2.4749 / 3.1798 / .5456 | 2.2954 / 2.9629 / .6055 | 2.3130 / 2.9472 / .6097 | **2.2820 / 2.9166 / .6177** |
| 2 | 2.4559 / 3.0924 / .5906 | 2.3576 / 3.0443 / .6032 | 2.2414 / 2.8792 / .6451 | **2.2334 / 2.8603 / .6498** |
| 3 | 2.4488 / 3.2093 / .5307 | 2.2401 / 2.9819 / .5948 | 2.2331 / 2.9381 / .6066 | **2.2202 / 2.9209 / .6112** |
| 4 | 2.2829 / 2.9242 / .5859 | 2.0736 / 2.6468 / .6608 | 2.0797 / 2.6009 / .6724 | **2.0621 / 2.5856 / .6763** |
| 5 | 2.3902 / 3.0724 / .4927 | 2.1486 / 2.7801 / .5846 | 2.1182 / 2.7308 / .5992 | **2.1109 / 2.7300 / .5995** |

Stacking had the lowest MAE and RMSE and highest R² in **all five** outer folds. Relative to LR, RF, and XGB respectively, its mean MAE was lower by **0.2288 (9.4917%)**, **0.0413 (1.8583%)**, and **0.0153 (0.6976%)**; mean RMSE was lower by **0.2929 (9.4626%)**, **0.0805 (2.7925%)**, and **0.0166 (0.5877%)**; mean R² was higher by **0.0818**, **0.0211**, and **0.0043**. These are descriptive differences, especially small versus XGB; no significance or uncertainty test of between-model differences is present. The fold means are unweighted means of equally sized folds, and SD reflects between-fold variation for this partition only.

## 8. Feature importance

Notebook 09 fits a fresh `ThesisPreprocessor` on **all 3,135 modeling counties**, retaining the same 14 features, then fits a separate RF and XGB on this full processed table. It hard-codes the modal outer-fold RF setting (`n_estimators=100`, `max_depth=20`, `min_samples_split=5`) and the XGB setting (`learning_rate=.05`, `n_estimators=200`, `max_depth=4`, `subsample=.8`), each with `random_state=42`, `n_jobs=-1`; XGB uses `objective="reg:squarederror"`. The code calls each estimator's `.feature_importances_`, sorts descending, and saves full and top-five tables. RF's measure is the normalized impurity-based tree importance (mean decrease in impurity). XGB's measure is **normalized gain**, specifically average gain across feature splits: notebook 09 does not set `importance_type`, and the installed `xgboost-3.4.1` implementation of `XGBRegressor.feature_importances_` calls `get_score(importance_type="gain")` for the default tree booster and normalizes scores to sum to one. This was verified in `.venv/lib/python3.12/site-packages/xgboost/sklearn.py` around lines 1641–1674, rather than inferred from documentation alone. These models were fitted on all records for description; their feature rankings and any in-sample training behavior are **not** nested-CV performance estimates.

| Rank | RF feature | RF importance | XGB feature | XGB importance |
|---:|---|---:|---|---:|
| 1 | MEDHHINC21 | .344833 | MEDHHINC21 | .369792 |
| 2 | PCT_18YOUNGER20 | .106171 | POVRATE21 | .094056 |
| 3 | PCT_NHBLACK20 | .092708 | PCT_18YOUNGER20 | .090547 |
| 4 | FSRPTH20 | .065290 | PCT_NHBLACK20 | .073678 |
| 5 | PCT_NHASIAN20 | .062301 | FSRPTH20 | .068083 |
| 6 | POVRATE21 | .059530 | PCT_NHASIAN20 | .064956 |
| 7 | PCT_HISP20 | .046183 | CONVSPTH20 | .043100 |
| 8 | CONVSPTH20 | .042747 | PCT_HISP20 | .040762 |
| 9 | PC_SNAPBEN22 | .037757 | PCT_65OLDER20 | .039056 |
| 10 | PCT_65OLDER20 | .036555 | PCT_NHWHITE20 | .030494 |
| 11 | PCT_NHWHITE20 | .029839 | PC_SNAPBEN22 | .028550 |
| 12 | PCT_LACCESS_LOWI19 | .028763 | FFRPTH20 | .022259 |
| 13 | FFRPTH20 | .024549 | DEEPPOVRATE21 | .018600 |
| 14 | DEEPPOVRATE21 | .022777 | PCT_LACCESS_LOWI19 | .016067 |

The common top-five predictors are `MEDHHINC21`, `PCT_18YOUNGER20`, `PCT_NHBLACK20`, and `FSRPTH20`. These scores describe predictive use in fitted tree models and are not evidence of causal effects. Correlated predictors can share or redistribute importance.

## 9. County predictions and mapping

`stacking_outer_predictions.csv` contains one held-out outer-fold stacking prediction per FIPS, as well as that fold's LR, RF and XGB base predictions, target and fold number. Notebook 10 joins the 3,135 records to USDA county/state names, calculates `Prediction_Difference = Stacking_Prediction - Actual`, then joins by five-character FIPS to the **2024 Census 1:500,000 county shapefile** `cb_2024_us_county_500k.shp`. A positive difference means overprediction, negative means underprediction, in percentage points. All 3,135 prediction FIPS match a unique boundary `GEOID`; none has missing geometry or target. The source boundary file has 3,235 geometries, leaving 100 without results. Geometry remains EPSG:4269. The saved difference range is **−9.6098 to +11.7833** percentage points.

The map CSV has seven attributes: `FIPS`, `State_Name`, `State_Abbreviation`, `County_Name`, `CDC_Obesity_AdjPrev`, `Predicted_Obesity_AdjPrev`, `Prediction_Difference`. The GeoJSON adds `geometry`. Notebook 10 created and displayed a Plotly Express choropleth colored by predicted prevalence, with county hover values and fitted bounds, but did not save an HTML figure or application. The CSV has no geometry; the GeoJSON has all 3,135 geometries. Map estimates represent validation predictions under five different fitted ensemble instances, not an all-data production fit.

## 10. Output files and intended manuscript uses

| Files | Content and use |
|---|---|
| `data/interim/modeling_data.csv` | Reproducible joined modeling table, with initial predictors and target; use for sample/variable audits |
| `outputs/linear_regression/lr_outer_fold_metrics.csv`, `lr_performance_summary.csv`, `lr_outer_predictions.csv` | LR fold results, mean/SD, held-out county predictions |
| `outputs/random_forest/rf_outer_fold_metrics.csv`, `rf_performance_summary.csv`, `rf_best_hyperparameters.csv`, `rf_outer_predictions.csv` | RF results, selected fold settings and predictions |
| `outputs/xgboost/xgb_outer_fold_metrics.csv`, `xgb_performance_summary.csv`, `xgb_best_hyperparameters.csv`, `xgb_outer_predictions.csv` | XGB results, selected fold settings and predictions |
| `outputs/stacking/stacking_outer_fold_metrics.csv`, `stacking_performance_summary.csv`, `stacking_best_hyperparameters.csv`, `stacking_meta_coefficients.csv`, `stacking_outer_predictions.csv` | Ensemble results, fold settings, meta coefficients and county OOF predictions; prediction file is map source |
| `outputs/model_comparison/model_performance_comparison.csv`, `all_model_outer_fold_metrics.csv`, `stacking_improvement.csv`, `best_model_by_fold.csv` | Primary manuscript performance table, fold comparison, descriptive improvement and fold winners |
| `outputs/feature_importance/rf_feature_importance.csv`, `xgb_feature_importance.csv`, `rf_top5_features.csv`, `xgb_top5_features.csv` | Full rankings and top-five figure inputs |
| `outputs/map/county_map_data.csv`, `county_map_data.geojson` | Seven map attributes and corresponding geometry for visualization; use GeoJSON for geospatial rendering |

The notebooks display model-comparison bars, feature-importance bars and a Plotly choropleth, but there are no saved PNG, PDF or HTML figure files in the output tree. `outputs/predictions/` and `outputs/metrics/` contain only placeholders.

## 11. Methodology–implementation consistency notes

1. The code uses **age-adjusted** CDC obesity prevalence, not crude prevalence, and county ecological predictors from different reference years. Source release date and target reference period should be described carefully rather than called one common study year.
2. The `-8888` rule excludes entire FIPS; `-9999` becomes missing and may later be imputed. The earlier inspection notebook's `missing_or_9999` percentage **does not count `-8888`** and should not be quoted as the final modeling missingness denominator.
3. Screening, imputation and correlation decisions are fitted within each model pipeline's training subset. The separate notebook 03 is a demonstration, not the preprocessing object passed to RF/XGB/stacking; the operational classes in notebooks 05–07 implement the same core rules. The stand-alone LR notebook defines its own equivalent function.
4. The correlation code records every |r|≥.80 pair but resolves only the two named pairs. Manuscript language such as “all highly correlated features were removed automatically” would overstate the implementation. IQR is diagnostic only; no observations were deleted or winsorized.
5. Standard scaling applies to LR alone. Default `LinearRegression` has no tuning and its stand-alone evaluation has no three-fold inner search. RF, XGB and stacking do have inner tuning. Thus “all four models underwent nested hyperparameter tuning” would be inaccurate.
6. Stacking uses a second, distinct three-fold OOF stage after parameter search. RF/XGB parameters are not reselected inside each stacking OOF training partition; disclose this if making a strict independence claim about meta-training rows. Outer validation remains separated.
7. Mean ± SD is across five folds of one seed, not repeated CV, a confidence interval, or a statistical test. The small advantage over XGB (0.0153 MAE; 0.0166 RMSE) warrants modest wording despite five fold wins.
8. Feature importance is from descriptive, full-data refits using hard-coded modal settings, not aggregated outer-fold importance. RF is impurity importance; XGB is normalized average gain. Neither supplies causal interpretation.
9. Map predictions are combined OOF estimates from five fold-specific models. The project does not save a final deployable stacked estimator fit on all 3,135 records. “Final model prediction” should be qualified accordingly.
10. The code uses random county folds rather than geographical or temporal holdouts; neighboring counties and shared state conditions may make generalization to wholly unseen regions different from these metrics.
11. Connecticut's old county FIPS and newer planning-region FIPS do not join; the final sample omits both unmatched sets. Report these geography changes in sample selection.
12. Notebook 04 uses a hard-coded Colab path (`/content/county-obesity-prediction`) whereas most other notebooks derive the project root from `Path.cwd().parent`; as written, fresh local execution of notebook 04 needs a path adjustment. Notebook 02 also includes a `pd.read_csv("modeling_data.csv")` cell with a file-not-found error, followed by a successful read from the correct interim path. These are execution/reproducibility issues, not evidence that the saved modeling file is absent.

## 12. Manuscript-ready fact sheet

**Dataset/sample:** 18 initial USDA predictors listed in section 2; 14 final predictors listed in section 3; 3,135 matched counties; target `OBESITY_AdjPrev` (CDC age-adjusted adult obesity prevalence). Missingness exclusions: `GROCPTH20` and `RECFACPTH20` (>20%; 28.92% and 58.83% in the pre-merge USDA wide table). Correlation exclusions: `CHILDPOVRATE21` and `PCT_LACCESS_POP19` (|Pearson r|≥.80 with their retained partners).

**Validation:** Five outer shuffled folds, 2,508/627 train/validation counties each; three inner shuffled folds for RF/XGB tuning; three stacking OOF folds; random seed 42. Metrics: MAE, RMSE, R²; means and sample SD across outer folds.

**Models:** LR: median imputation, screens, standardization, default OLS. RF: screens/imputation, random forest, 10-draw random search. XGB: screens/imputation, squared-error boosted trees, 10-draw random search. Stacking: LR/RF/XGB base pipelines, OOF training predictions, LR meta-learner.

**Final performance (MAE; RMSE; R², mean ± SD):** LR **2.4105 ± .0781; 3.0956 ± .1117; .5491 ± .0407**. RF **2.2231 ± .1135; 2.8832 ± .1648; .6098 ± .0297**. XGB **2.1971 ± .0958; 2.8193 ± .1497; .6266 ± .0311**. Stacking **2.1817 ± .0916; 2.8027 ± .1438; .6309 ± .0315**.

**Feature importance:** RF top five: `MEDHHINC21`, `PCT_18YOUNGER20`, `PCT_NHBLACK20`, `FSRPTH20`, `PCT_NHASIAN20`. XGB top five: `MEDHHINC21`, `POVRATE21`, `PCT_18YOUNGER20`, `PCT_NHBLACK20`, `FSRPTH20`. Shared: `MEDHHINC21`, `PCT_18YOUNGER20`, `PCT_NHBLACK20`, `FSRPTH20`.

**Mapping:** 3,135 counties; source is held-out outer-fold stacking predictions. Main files: `outputs/model_comparison/model_performance_comparison.csv`, `outputs/model_comparison/all_model_outer_fold_metrics.csv`, `outputs/feature_importance/{rf,xgb}_feature_importance.csv`, `outputs/stacking/stacking_outer_predictions.csv`, `outputs/map/county_map_data.{csv,geojson}`.

## 13. Manuscript-ready methodology

We assembled county-level predictors from the USDA Food Environment Atlas and age-adjusted adult obesity prevalence (`OBESITY_AdjPrev`) from the CDC PLACES 2024 county release. We selected 18 USDA variables covering food access, food retail and restaurants, SNAP benefits, income, poverty, age and race/ethnicity composition. We formatted USDA and CDC FIPS as five-character strings and joined by FIPS. USDA FIPS containing a `-8888` value in any selected predictor were excluded, while `-9999` values were coded missing. We pivoted the USDA records to one row per FIPS and inner-joined them with the CDC target, producing 3,135 counties. Unmatched Connecticut county and planning-region codes were not reconciled.

We evaluated ordinary least squares linear regression, random forest regression, XGBoost regression, and a stacking ensemble using five-fold shuffled outer cross-validation with seed 42. Each outer split contained 2,508 training and 627 held-out validation counties. Within each training fit, predictors with more than 20% missing values were removed and remaining missing entries were median-imputed. Pearson correlations were computed after imputation; when |r| was at least .80, predefined rules removed child poverty rate in favor of poverty rate and general low-access percentage in favor of low-income low-access percentage. We calculated 1.5-IQR outlier counts but did not delete or alter flagged observations. Linear regression inputs were standardized with training-fitted means and variances; tree models used unscaled imputed inputs. All preprocessing decisions were re-estimated within the relevant training subset.

Random forest `n_estimators`, `max_depth`, and `min_samples_split` and XGBoost `learning_rate`, `n_estimators`, `max_depth`, and `subsample` were selected by 10-iteration randomized searches using three shuffled inner folds, seed 42, and negative RMSE. The searches included preprocessing in each candidate pipeline and refitted the selected pipeline on the outer training set. The linear model used default `LinearRegression` without a hyperparameter search. In each outer fold, stacking tuned RF and XGB on the outer training set, then generated three-fold OOF predictions from LR, RF and XGB base pipelines using those settings. A linear regression meta-learner was fitted to the three OOF prediction columns. The base pipelines were refitted on the full outer training set and predicted the untouched outer validation fold; the meta-learner combined those three values into the final fold prediction. We calculated MAE, RMSE and R² for each outer fold and summarized their five values with arithmetic mean and sample SD.

For descriptive feature rankings, we separately fitted the preprocessing procedure and RF/XGB models on all 3,135 counties using modal outer-fold parameter settings, then obtained normalized impurity importance from RF and normalized average gain importance from XGB. These refits were not used to estimate held-out performance. For mapping, we joined each county's held-out stacking prediction and observed target to 2024 Census county geometry by FIPS and computed predicted minus observed obesity prevalence in percentage points.

## 14. Manuscript-ready results

**Observed results.** The joined sample had 3,135 counties. The more-than-20% missingness rule excluded grocery-store density and recreation-facility density in every outer fold. Two specified correlation exclusions left 14 predictors in each outer fold. Across five held-out folds, stacking achieved MAE **2.1817 ± 0.0916**, RMSE **2.8027 ± 0.1438**, and R² **0.6309 ± 0.0315**. XGB was the strongest individual model (MAE **2.1971 ± 0.0958**, RMSE **2.8193 ± 0.1497**, R² **0.6266 ± 0.0311**). Stacking ranked first on all three metrics in every fold, but its mean advantage over XGB was **0.0153 MAE**, **0.0166 RMSE**, and **0.0043 R²**. The all-data descriptive RF and XGB feature rankings both placed median household income first and shared four top-five predictors. All 3,135 OOF stacking predictions matched county geometry.

**Reasonable interpretation.** The ensemble had the best observed predictive performance under this random county-level partition. Its improvement over the strongest single learner was small and has not been tested for statistical significance. Feature importance indicates which variables the fitted tree models used most for prediction under the chosen preprocessing and hyperparameters; it does not establish the direction or cause of any relation with obesity. The county map shows held-out prediction error and predicted prevalence within the evaluated sample; it is not a prospective estimate for counties outside the sample.

## 15. Items I should verify manually

1. Confirm the CDC file's exact publication citation, data vintage, age-adjustment definition and target reference years from authoritative source metadata before publication; the repository filename identifies the 2024 release and an August 2026 local filename date but does not establish every epidemiologic detail.
2. Confirm USDA Food Environment Atlas citation, variable definitions, and units. `VariableList.csv` labels `FSRPTH20` as “Count” although its name describes restaurants per 1,000 population; use the source documentation to resolve that unit discrepancy. Likewise verify how the selected special codes are formally defined by USDA; this report states how the code handled them.
3. Decide how to explain the exclusion of Connecticut old county FIPS and planning-region FIPS, plus four other `-8888` FIPS, in a sample-flow figure. Verify whether a geographic crosswalk would materially change the intended study population before any reanalysis.
4. Verify the exact frozen dependency versions and computational environment for reproducibility; no project requirements or lockfile was found. This report's XGB gain conclusion was confirmed against the installed `xgboost-3.4.1` source and should be revisited if results are regenerated under another version.
5. Confirm that saved CSVs correspond to the intended final notebook execution. The model-comparison files agree with model-specific saved metrics, and the feature and map files match their notebook outputs, but the project has no run manifest, checksums or serialized fitted model.
6. Fix the hard-coded Colab path in notebook 04 and the erroneous intermediate CSV read in notebook 02 before claiming a clean local start-to-finish rerun. No notebook rerun was performed for this read-only review.
7. If publication claims uncertainty in model differences, conduct and report a suitable preplanned comparison; no statistical significance test, confidence interval, repeated CV or spatial validation is implemented.
8. If the paper describes a deployable county prediction system, specify how one final model would be fitted and versioned. The saved map currently uses five held-out fold models, and no single production stack is saved.

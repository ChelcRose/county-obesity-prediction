## USDA Raw Data Inspection

- The USDA Food Environment Atlas contains 957,753 rows and 5 columns.
- The dataset is in long format with the columns FIPS, State, County, Variable_Code, and Value.
- It contains 304 unique variable codes and 3,156 unique non-missing FIPS values.
- FIPS is read as a float (`float64`), causing leading zeroes to be omitted (e.g., `01001` appears as `1001.0`). FIPS will be standardized to five-digit strings during data integration.
- 475 rows have missing FIPS values.
- All 475 missing-FIPS rows correspond to five Alaska area records: Aleutian Islands Area, Anchorage Area, Fairbanks Area, Juneau Area, and Kenai Peninsula Area.
- Each of these areas contains 95 variable records.
- None of the 475 missing-FIPS rows correspond to the 18 predictors selected for the study.


### Special-Value Inspection

- Among the 18 selected predictors, there are no ordinary NaN values in the raw `Value` column.
- There are 3,855 occurrences of `-9999`, which USDA defines as missing or unavailable data.
- There are 111 occurrences of `-8888`, representing nonexistence of the measured entity.
- The 111 `-8888` values occur across 13 unique county FIPS codes.
- Under the study methodology, counties containing `-8888` in any required predictor will be excluded during preprocessing.
- No duplicate predictor-FIPS combinations were found.
- Predictor coverage differs across indicators, ranging from 3,144 to 3,153 county records among most selected predictors, with `PC_SNAPBEN22` containing 3,147 records.

### Descriptive Missingness Inspection

Missingness was inspected descriptively across the complete raw dataset for the 18 selected predictors. This inspection is exploratory only and does not determine final feature exclusion, since the study methodology requires the >20% missingness rule to be applied using the training portion of each outer fold.

- GROCPTH20: 28.91% missing/-9999
- RECFACPTH20: 58.81% missing/-9999
- All other selected predictors showed less than 20% descriptive missingness.
- Final missingness-based feature exclusion will be determined independently within each outer training portion during nested cross-validation.

## CDC PLACES Raw Data Inspection

- The CDC PLACES 2024 GIS-friendly dataset contains 3,144 rows and 167 columns.
- `OBESITY_AdjPrev`, the target variable specified for the study, is present and stored as a numeric (`float64`) variable.
- The dataset contains 3,144 unique CountyFIPS values, with no duplicate or missing CountyFIPS values.
- No missing values were found in `OBESITY_AdjPrev`.
- `OBESITY_AdjPrev` ranges from 17.7% to 53.0%, with a mean of approximately 37.91%.
- `CountyFIPS` is read as an integer (`int64`), causing leading zeroes to be omitted for some counties (e.g., `02020` appears as `2020`). County FIPS will be standardized to five-digit strings before dataset integration.
- No `OBESITY_AdjPrev` values were outside the valid percentage range of 0% to 100%.
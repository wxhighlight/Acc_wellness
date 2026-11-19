# Data and code

The tables cover 3,734,675 populated grid cells and 2,850 county-level units in mainland China, representing 1,410,585,044 residents. The grid and county tables use the 120-minute driving window. The scenario table reports the alternative windows and sensitivity cases.

## Data

| File | Records | Contents |
| --- | ---: | --- |
| `Data/Grid_accessibility.csv.gz` | 3,734,675 | Grid coordinates, population, need, and accessibility |
| `Data/County_equity.csv` | 2,850 | County indicators, local cluster results, and supply-demand measures |
| `Data/Sensitivity_results.csv` | 81 | Scenario estimates and 95% confidence intervals |

`eco`, `psy`, `med`, and `C` denote ecological healing, mental restoration, medical care, and composite accessibility. `population` is the modeled 2025 resident count. `wni` is the wellness need index on the 0–1 scale. `r_eco`, `r_psy`, and `r_med` are divided by their respective population-weighted national means. `C` is the geometric mean after adding 0.001 to each dimension. A zero in any of the three single-resource columns identifies a cell without access to at least one resource.

### Grid coordinates

`row` and `col` are zero-based indices on the nominal 1 km model grid (EPSG:4326, 7,392 columns by 5,968 rows). Use the following formulas for cell centers in decimal degrees.

```text
longitude = 73.49896199959676 + (col + 0.5) * 0.008332097352482548
latitude  = 53.558497999882945 - (row + 0.5) * 0.008332097352482550
```

Rows follow grid order. `county_id` links each row to the county table. Coordinates and resident counts retain their original integer precision. WNI and accessibility are stored to seven significant digits.

### County fields

| Field | Definition |
| --- | --- |
| `county_id`, `admin_code`, `province_id` | Study county identifier, six-digit administrative code, and province identifier |
| `wni`, `r_*`, `C` | Population-weighted grid values within the county |
| `income`, `age60_share` | 2024 per capita disposable income in CNY and the fraction aged 60 or over |
| `match_*` | Same-county need–access type, with need first and accessibility second. The four types are `HH`, `HL`, `LH`, and `LL` |
| `cluster_*`, `upper_tail_*` | Significant cluster type (`NS` for nonsignificant) and number of draws at or above the observed local join count |
| `neighbor_ids` | Pipe-separated `county_id` values for the binary Queen neighbors. Isolated counties use eight nearest neighbors |
| `V_*`, `K_*` | Accessible supply score and effective sharing population, defined as accessible supply divided by raw accessibility |
| `scarcity_*`, `competition_*`, `structure_*` | Signed log contributions and dominant component (`scarcity`, `competition`, or `zero`) for the resource’s significant H-L counties |

Blank decomposition fields identify counties outside the relevant H-L set, or undefined log terms at zero access. `K_*` is undefined at zero raw accessibility. The composite contributes to the four-type cluster analysis. The supply-demand decomposition applies to the three single resources.

### Scenario fields

`scenario` and `driving_minutes` identify the case. The cases include the baseline, the 60- and 90-minute windows, cold-season availability, village scores of 10 and 50 (baseline 30), traditional Chinese medicine health-tourism base scores of 30 and 60 (baseline 50), and arithmetic aggregation.

`metric`, `resource`, and `group` identify each estimate. `need_gradient` is the Q5 minus Q1 accessibility difference. `income_concentration` is the national composite income concentration index. `high_need_income_gap` subtracts high-income accessibility from low-income accessibility in high-need counties. `zero_access_pct` uses the resident population within the specified group as denominator, and `zero_access_gap_pp` reports the difference in percentage points. Resource `any` means zero raw accessibility to at least one resource. `estimate`, `ci_low`, and `ci_high` give the point estimate and its 95% interval.

## Run with the supplied tables

Use Python 3.9 with the versions in `Code/requirements.txt`. Run from this directory.

```sh
python -m pip install -r Code/requirements.txt
python Code/04_Equity/equity.py
```

The equity script writes grid need gradients, county concentration estimates, the need-income matrix, income gaps, co-location results, and H-L decomposition to `outputs/equity`. Add `--rerun-permutations` to regenerate the 10,000 draws. The default calculates p and q from the saved counts.

Need gradients and the need-income matrix use population-weighted quintiles. Within-need concentration indices rank counties again by income within each need group. The map types use the national population-weighted WNI mean and the corresponding accessibility reference (1 for `r`, the national population-weighted mean for `C`). BH adjusts each resource/type family over 2,850 counties using `(upper_tail + 1) / 10001` for members and 1 for nonmembers, with `q < 0.05`. Supply-demand shares use each resource’s significant H-L population. The travel-time comparisons use county sample medians for need and income and county sample terciles for ageing. Equity and scenario intervals use 1,000 province bootstrap samples with seed 42.

## Run with aligned model inputs

The input contract starts at the aligned grid and scored source cells. Build WNI and calibrate the time thresholds before running 2SFCA. The equity command adds component gradients to the supplied baseline analysis. The sensitivity command reads the modeled scenario arrays.

```sh
python Code/01_WNI/build_need_index.py --grid Aligned/grid_covariates.npz --output outputs/need_index.npz
python Code/02_Travel_time/calibrate.py --pairs Aligned/travel_time_pairs.csv --output outputs/calibration
python Code/03_Accessibility/accessibility.py --inputs Aligned --output outputs/model
python Code/04_Equity/equity.py --aligned-grid Aligned/grid_covariates.npz
python Code/05_Sensitivity/sensitivity.py --model outputs/model --grid Aligned/grid_covariates.npz
```

- `grid_covariates.npz` uses one common grid row order. `row`, `col`, and `population` support 2SFCA. `pm25`, `tsi`, and `age60_share` support WNI. WNI equally averages the indicators after population-weighted 1st–99th percentile clipping and scaling to 0–1. Use the resulting `wni` and matching `county_id` in the aligned grid for the scenario script. `heat_stress` and `cold_stress` provide the additional component gradients in step 4.
- `travel_time_pairs.csv` has `friction_minutes`, `amap_minutes`, `baidu_minutes`, and `tencent_minutes` for matched origin–destination pairs. Calibration searches 20–140 friction minutes by one-minute steps and minimizes `2 × false positives + false negatives`. The bootstrap uses 500 resamples. The displayed segmented fit uses positive pairs below 180 minutes on both axes and the 120-minute window’s threshold as its breakpoint.
- `friction_minutes_per_metre.tif` uses the model grid. `resource_sources.csv` has one row per scored source cell with zero-based `row` and `col`, the `cold_stress` month fraction, and `S_` score fields matching `COMPONENTS` in `accessibility.py`. The travel-path kernel and model use eight neighbors, endpoint-mean friction, and latitude-dependent step lengths. Driving windows of 60, 90, and 120 minutes use calibrated friction thresholds of 41, 70, and 100 minutes, respectively. The Gaussian decay coefficient is 1.5 and the weighted demand floor is 1,000 residents.
- Scenario reruns use the baseline, travel-window, and sensitivity NPZ files written in step 3. The cold-season case scales ecological and mental-restoration supply by `1 − cold_stress`. The script also uses the component arrays to adjust village and TCM scores.

## Source data

- Population and age structure use the 2025 China layers from the WorldPop R2025A releases. [Population counts](https://hub.worldpop.org/doi/10.5258/SOTON/WP00840), [age and sex](https://hub.worldpop.org/doi/10.5258/SOTON/WP00842).
- PM2.5 uses annual ChinaHighPM2.5 maps for 2020–2024. Thermal stress uses the 2015–2015 monthly GloUTCI-M inputs, with heat above 26 °C and cold below 9 °C. [ChinaHighPM2.5](https://weijing-rs.github.io/product.html), [GloUTCI-M v1 & v2](https://zenodo.org/records/8310513).
- Income is aggregated to counties from the 2024 downscaled per capita disposable income surface described by Wang et al. [Income source](https://doi.org/10.1038/s41597-026-07188-1).
- The friction surface combines road classes, terrain, and land cover. Map-service driving times are matched by origin and destination. [OpenStreetMap](https://www.openstreetmap.org/), [Geospatial Data Cloud](https://www.gscloud.cn/), [Amap](https://lbs.amap.com/), [Baidu Maps](https://lbsyun.baidu.com/), [Tencent Maps](https://lbs.qq.com/).
- The scored source inventory uses government registers of attractions, forest and wetland parks, villages, medical facilities, TCM health-tourism bases, and elder-care facilities. [Culture and Tourism](https://www.mct.gov.cn/), [Forestry and Grassland](https://www.forestry.gov.cn/), [Housing and Urban-Rural Development](https://www.mohurd.gov.cn/), [Healthcare Security](https://fuwu.nhsa.gov.cn/), [Traditional Chinese Medicine](https://www.natcm.gov.cn/), [Elder-care Services](https://yanglao.mca.gov.cn/).

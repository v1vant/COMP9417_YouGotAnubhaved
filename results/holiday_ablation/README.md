# Holiday-context demand ablation

Run from the repository root in the project Python environment:

```bash
python code/ablate_holiday_context.py
```

This compares standard LightGBM demand forecasting with and without three added
calendar flags: NSW holiday yesterday, NSW holiday tomorrow, and 20 December–5
January inclusive. Both variants retain the existing public-holiday indicator.
Model settings and expanding-history replacement dates are identical.
The initial fit ends in 2014; replacements start in 2016, 2018, 2020, 2022 and 2024.
All reported results are from development market days in 2015–2024.
This experiment was prompted by inspecting development errors and is exploratory.

| Group | Baseline MAPE (%) | Added context MAPE (%) |
| --- | ---: | ---: |
| All | 2.594458 | 2.542052 |
| Public holidays | 3.565808 | 3.357242 |
| Adjacent nonholidays | 3.723046 | 3.091225 |
| Christmas window | 3.908904 | 3.720895 |
| Ordinary days | 2.485347 | 2.462347 |
| Midday | 4.510880 | 4.428342 |
| Evening | 2.240655 | 2.178091 |

Overall MAE decreases from 200.33 MW to 196.41 MW. Overall and adjacent-nonholiday
MAPE improve in all ten years; ordinary-day and Christmas-window MAPE improve
in nine years. Groups overlap; ordinary excludes holidays, adjacent days and
the Christmas window. The baseline exactly matches the saved development score
in the parent directory's aggregate_metrics.csv.

The exploratory paired-day bootstrap interval for overall MAPE change is
[-0.062258, -0.043204] percentage points. It does not adjust for choosing this
experiment after error inspection or all possible serial dependence between
days. The three features were tested together; their individual effects are
not isolated.

Individual dates can worsen: 23 December 2024 changes from 14.751% to 15.201%
MAPE, while 24 December improves from 22.066% to 19.783%.

metrics.csv contains aggregate and yearly results; daily_paired_errors.csv
contains date-level comparisons; paired_deltas.csv contains bootstrap summaries;
fits.csv records training cutoffs; manifest.json records parameters and hashes.
yearly_deltas.csv is a supplementary pivot of metrics.csv, with negative changes
favouring the added context. No 2025–2026 evaluation was run, and final trained
artifacts and default configurations were not replaced.

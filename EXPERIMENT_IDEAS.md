# Deferred experiment ideas

- Quantile price regression: predict lower, median, and upper NSW prices, then
  provide their margins against the previous day's mean and the interval width
  to an augmented classifier. Generate classifier-training inputs out of sample
  using chronological folds. This candidate is not implemented or evaluated.
- Reference-day-centred price profiles: subtract each historical day's own mean
  to isolate intraday shape, alongside the current D-relative margins. Deferred.

Keep the current validated 405-demand/571-price model unchanged while evaluating
these separate candidates. Use pre-holdout periods for development; the existing
2025–2026 holdout has already been inspected for the current model.

"""Compare holiday-enabled demand LightGBM with/without solar trend features.

Uses fixed settings and periodic expanding-history retraining on development
market days 2015-2024. Price classification and final artifacts are unchanged.
"""
from threadpoolctl import threadpool_limits

from ablate_holiday_context import main


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main(feature_set="solar_trend")

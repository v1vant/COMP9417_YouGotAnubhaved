"""Shared helpers for the day-type scripts in this folder.

Data are read with nrows from the start of the repo CSV, so rows after the requested market day are never loaded.
Figures that go past 2014 run to the end of the data (END = 2026-10-31), including 2025-2026; these are descriptive
day-type figures, not model testing, and nothing is chosen or tuned from them. Clustering uses the team's day_type_analysis in code/analysis_tools.py
(shape = day / day mean; PCA 95% + k-means, both fitted on 2010-2013 only)."""
import os, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
CSV = os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")
OUT = os.path.join(ROOT, "results", "day_types_2010_2026")
END = "2026-10-31"  # last market day in the data
sys.path.insert(0, CODE)
import analysis_tools as team  # noqa: E402  (the team's day-type code, unmodified)

COLOURS = ["#147D92", "#8A58A0"]
NAMES = ["Type 0 · evening peak", "Type 1 · daytime-heavy"]


def load_demand(last_day):
    """NSW demand indexed by interval-end timestamp, market days 2010-01-01 .. last_day inclusive."""
    n_days = (pd.Timestamp(last_day) - pd.Timestamp("2010-01-01")).days + 1
    frame = pd.read_csv(CSV, usecols=["timestamp", "nsw_demand"], parse_dates=["timestamp"],
                        nrows=n_days * 48).set_index("timestamp")
    assert frame.index[0] == pd.Timestamp("2010-01-01 00:30")
    assert frame.index[-1] == pd.Timestamp(last_day) + pd.Timedelta(days=1), "rows do not end at the requested day"
    return frame


def team_day_types(last_day=END):
    types, shares, typical, info = team.day_type_analysis(load_demand(last_day))
    assert info["reference"] == "2010-2013" and info["selected_clusters"] == 2
    return types, shares, typical, info


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="day_types_"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#344054', 'text.color': '#182230', 'xtick.color': '#475467', 'ytick.color': '#475467',
        'axes.edgecolor': '#D0D5DD', 'legend.frameon': False, 'savefig.facecolor': 'white'})
    return plt


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    print("saved", os.path.join(OUT, name + ".png"))

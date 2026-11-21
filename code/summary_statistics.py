"""Summary statistics for the report: one compact table and one yearly-trend figure.

Descriptive only: nothing is fitted or selected, so using all years does not touch the
holdout discipline.

Usage (from the project root):
    python code/summary_statistics.py
    python code/summary_statistics.py --data data/nem_weather_halfhourly_2010_2026.csv --out results/summary_stats

Outputs in --out:
    summary_table.tex      LaTeX table (booktabs), whole period, plus the 2010-14 vs 2015-26 means
    summary_table.csv      the same numbers
    yearly_stats.csv, monthly_stats.csv, hourly_stats.csv  the numbers behind the figure
    concept_drift.pdf/png  demand under similar weather: by time of day and year, and midday demand vs temperature
    key_trends.pdf/png     1x4: demand by year, demand by hour of day (2010-14 vs 2015-26), price by year, share of price-up half-hours
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# (column, label in the table, unit)
VARIABLES = [
    ("nsw_demand", "NSW demand", "MW"),
    ("nsw_price", "NSW price", "\\$/MWh"),
    ("syd_temp", "Sydney temperature", "$^\\circ$C"),
    ("syd_rh", "Sydney humidity", "\\%"),
    ("syd_solar", "Sydney solar", "W/m$^2$"),
]


def market_days(index):
    """Market day D covers 00:30 on D to 00:00 on D+1 (same rule as harness/dayahead.py)."""
    return (index - pd.Timedelta(minutes=30)).normalize()


def load(path):
    return pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"]).sort_index()


def price_up(frame):
    """1 where the half-hour price exceeds the previous market day's mean NSW price."""
    days = market_days(frame.index)
    daily_mean = frame["nsw_price"].groupby(days).mean()
    previous = pd.Series(days - pd.Timedelta(days=1), index=frame.index).map(daily_mean)
    up = (frame["nsw_price"] > previous).astype(float)
    return up.where(previous.notna())  # first day has no threshold


def table(frame):
    rows = []
    early = frame.index < pd.Timestamp("2015-01-01")
    for col, label, unit in VARIABLES:
        x = frame[col]
        rows.append({
            "variable": label, "unit": unit,
            "mean": x.mean(), "median": x.median(), "std": x.std(),
            "min": x.min(), "max": x.max(),
            "mean_2010_14": x[early].mean(), "mean_2015_26": x[~early].mean(),
        })
    return pd.DataFrame(rows)


def latex(df):
    def fmt(v):
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}"
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\begin{tabular}{llrrrrrrr}", "\\toprule",
        "Variable & Unit & Mean & Median & Std & Min & Max & Mean 2010--14 & Mean 2015--26 \\\\",
        "\\midrule",
    ]
    for r in df.itertuples():
        values = " & ".join(fmt(v) for v in (r.mean, r.median, r.std, r.min, r.max,
                                             r.mean_2010_14, r.mean_2015_26))
        lines.append(f"{r.variable} & {r.unit} & {values} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Summary statistics of half-hourly observations, Jan 2010 to Oct 2026 "
              "(295{,}104 rows). Price is heavy-tailed: its mean is far above its median.}",
              "\\label{tab:summary}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def yearly(frame):
    year = market_days(frame.index).year
    up = price_up(frame)
    out = pd.DataFrame({
        "demand_mean": frame["nsw_demand"].groupby(year).mean(),
        "demand_p05": frame["nsw_demand"].groupby(year).quantile(0.05),
        "demand_p95": frame["nsw_demand"].groupby(year).quantile(0.95),
        "demand_q25": frame["nsw_demand"].groupby(year).quantile(0.25),
        "demand_q75": frame["nsw_demand"].groupby(year).quantile(0.75),
        "price_q25": frame["nsw_price"].groupby(year).quantile(0.25),
        "price_q75": frame["nsw_price"].groupby(year).quantile(0.75),
        "price_median": frame["nsw_price"].groupby(year).median(),
        "price_mean": frame["nsw_price"].groupby(year).mean(),
        "price_p05": frame["nsw_price"].groupby(year).quantile(0.05),
        "price_p95": frame["nsw_price"].groupby(year).quantile(0.95),
        "price_up_share": up.groupby(year).mean(),
        "rows": frame["nsw_price"].groupby(year).size(),
    })
    out.index.name = "year"
    return out


def profile(frame, key, label):
    """Median/mean/quartiles of demand and price by `key`, split into two periods."""
    early = market_days(frame.index) < pd.Timestamp("2015-01-01")
    parts = []
    for name, mask in (("2010-14", early), ("2015-26", ~early)):
        sub, k = frame[mask], key[mask]
        g = sub.groupby(k)
        out = pd.DataFrame({
            "demand_mean": g["nsw_demand"].mean(),
            "demand_q25": g["nsw_demand"].quantile(0.25),
            "demand_q75": g["nsw_demand"].quantile(0.75),
            "price_median": g["nsw_price"].median(),
            "price_mean": g["nsw_price"].mean(),
            "price_q25": g["nsw_price"].quantile(0.25),
            "price_q75": g["nsw_price"].quantile(0.75),
        })
        out.insert(0, "period", name)
        out.index.name = label
        parts.append(out)
    return pd.concat(parts)


def hourly_overall(frame):
    """Demand by clock hour of day (interval start), split into 2010-14 and 2015-26."""
    clock = frame.index - pd.Timedelta(minutes=30)
    early = clock.year < 2015
    parts = {}
    for name, mask in (("2010-14", early), ("2015-26", ~early)):
        g = frame.loc[mask, "nsw_demand"].groupby(clock[mask].hour)
        parts[name] = pd.DataFrame({"demand_mean": g.mean(), "demand_q25": g.quantile(0.25),
                                    "demand_q75": g.quantile(0.75)})
    return parts


def figure(stats, hours, path_stem):
    """1x4: demand by year, demand by hour of day, price by year, share of price-up half-hours."""
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
    x = stats.index
    colour, red = "#2a6f97", "#bc4b51"

    ax = axes[0]
    ax.fill_between(x, stats["demand_q25"], stats["demand_q75"], color=colour, alpha=0.18,
                    label="25th-75th pct")
    ax.plot(x, stats["demand_mean"], color=colour, marker="o", ms=3, label="mean")
    ax.set(title="NSW demand (MW)", xlabel="year")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    ax = axes[2]
    ax.fill_between(x, stats["price_q25"], stats["price_q75"], color=colour, alpha=0.18,
                    label="25th-75th pct")
    ax.plot(x, stats["price_median"], color=colour, marker="o", ms=3, label="median")
    ax.plot(x, stats["price_mean"], color=red, marker="s", ms=3, ls="--", label="mean")
    ax.set(title="NSW price ($/MWh)", xlabel="year")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    ax = axes[1]
    for name, line in (("2010-14", red), ("2015-26", colour)):
        d = hours[name]
        ax.fill_between(d.index, d["demand_q25"], d["demand_q75"], color=line, alpha=0.14, lw=0)
        ax.plot(d.index, d["demand_mean"], color=line, marker="o", ms=3, label=f"{name} mean")
    ax.fill_between([], [], [], color="grey", alpha=0.2, label="25th-75th pct")
    ax.set(title="NSW demand by hour of day (MW)", xlabel="hour of day")
    ax.set_xticks(range(0, 24, 4))
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    ax = axes[3]
    ax.plot(x, stats["price_up_share"], color=colour, marker="o", ms=3)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set(title="Share of half-hours with price above\nprevious day's mean", xlabel="year",
           ylim=(0.3, 0.6))

    for ax in (axes[0], axes[2], axes[3]):
        ax.set_xticks(range(x.min(), x.max() + 1, 4))
    for ax in axes.ravel():
        ax.grid(alpha=0.25, lw=0.5)
    fig.text(0.999, 0.005, "2026 ends 31 Oct", ha="right", va="bottom", fontsize=7, color="grey")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200)
    plt.close(fig)


def drift_tables(frame):
    """Demand under (nearly) fixed weather, by time of day and year. Controls: weekdays, Sydney
    temperature 18-22 C, humidity 60-80%, and solar radiation inside the middle half (25th-75th
    percentile) of that time band's values. Also: midday weekday demand against temperature for
    three eras."""
    clock = frame.index - pd.Timedelta(minutes=30)
    d = frame[["nsw_demand", "syd_temp", "syd_rh", "syd_solar"]].copy()
    d["year"], d["hour"], d["weekday"] = clock.year, clock.hour, clock.dayofweek < 5
    d = d[d["weekday"]]
    bands = {"midday (11:00-15:00)": (11, 15), "evening (17:00-20:00)": (17, 20),
             "night (01:00-05:00)": (1, 5)}
    rows = []
    for name, (h0, h1) in bands.items():
        sel = d[(d["hour"] >= h0) & (d["hour"] < h1) & d["syd_temp"].between(18, 22, inclusive="left")]
        lo, hi = sel["syd_solar"].quantile([0.25, 0.75])
        sel = sel[sel["syd_solar"].between(lo, hi) & sel["syd_rh"].between(60, 80)]
        g = sel.groupby("year")["nsw_demand"].agg(["mean", "size"])
        g.insert(0, "band", name)
        g["solar_lo"], g["solar_hi"] = lo, hi
        rows.append(g)
    by_year = pd.concat(rows)
    midday = d[(d["hour"] >= 11) & (d["hour"] < 15)].copy()
    midday["bin"] = (np.floor(midday["syd_temp"] / 2) * 2 + 1).astype(int)  # bin centre
    eras = {"2010-14": (2010, 2014), "2015-19": (2015, 2019), "2020-26": (2020, 2026)}
    curves = []
    for name, (y0, y1) in eras.items():
        sel = midday[midday["year"].between(y0, y1)]
        g = sel.groupby("bin")["nsw_demand"].agg(["mean", "size"])
        g = g[g["size"] >= 30]
        g.insert(0, "era", name)
        curves.append(g)
    return by_year, pd.concat(curves)


def drift_figure(by_year, curves, path_stem):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.92))  # room for the footnote
    blue, red, grey = "#2a6f97", "#bc4b51", "#8b949e"

    ax = axes[0]
    for (name, colour) in (("midday (11:00-15:00)", blue), ("evening (17:00-20:00)", red),
                           ("night (01:00-05:00)", grey)):
        d = by_year[by_year["band"] == name]
        lo, hi = d["solar_lo"].iloc[0], d["solar_hi"].iloc[0]
        solar = "solar 0" if hi < 1 else f"solar {lo:.0f}-{hi:.0f} W/m$^2$"
        d = d[d["size"] >= 30]  # skip years with too few matching half-hours
        ax.plot(d.index, d["mean"], color=colour, marker="o", ms=3,
                label=f"{name}, {solar}")
    ax.set(title="Demand at similar weather, by year\n(weekdays, 18-22 \u00b0C, humidity 60-80%)",
           xlabel="year", ylabel="mean NSW demand (MW)", ylim=(5000, None))
    ax.set_xticks(range(2010, 2027, 4))
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    ax = axes[1]
    colours = {"2010-14": red, "2015-19": "#8fb8d0", "2020-26": blue}
    for era, colour in colours.items():
        d = curves[curves["era"] == era]
        ax.plot(d.index, d["mean"], color=colour, marker="o", ms=3, label=era)
    ax.set(title="Midday demand vs temperature, by era\n(weekdays, 11:00-15:00)",
           xlabel="Sydney temperature (\u00b0C)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    fig.text(0.5, 0.005, "Left: solar range in each legend entry is the middle half (25th-75th percentile) of that "
             "time band's solar at 18-22 \u00b0C; years with fewer than 30 matching half-hours omitted.\n"
             "2026 ends 31 Oct. Public holidays are not excluded.",
             ha="center", va="bottom", fontsize=7, color="grey")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/nem_weather_halfhourly_2010_2026.csv")
    parser.add_argument("--out", default="results/summary_stats")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frame = load(args.data)
    summary = table(frame)
    summary.to_csv(os.path.join(args.out, "summary_table.csv"), index=False)
    with open(os.path.join(args.out, "summary_table.tex"), "w") as handle:
        handle.write(latex(summary))

    stats = yearly(frame)
    stats.to_csv(os.path.join(args.out, "yearly_stats.csv"))

    clock = frame.index - pd.Timedelta(minutes=30)  # interval start
    monthly = profile(frame, pd.Series(market_days(frame.index).month, index=frame.index), "month")
    hourly = profile(frame, pd.Series(clock.hour, index=frame.index), "hour")
    monthly.to_csv(os.path.join(args.out, "monthly_stats.csv"))
    hourly.to_csv(os.path.join(args.out, "hourly_stats.csv"))
    by_year, curves = drift_tables(frame)
    by_year.to_csv(os.path.join(args.out, "drift_by_year.csv"))
    curves.to_csv(os.path.join(args.out, "drift_midday_curves.csv"))
    drift_figure(by_year, curves, os.path.join(args.out, "concept_drift"))
    figure(stats, hourly_overall(frame), os.path.join(args.out, "key_trends"))

    pd.set_option("display.float_format", lambda v: f"{v:,.2f}", "display.width", 200)
    print(summary.drop(columns="unit").to_string(index=False))
    print()
    print(stats.round(3).to_string())
    print(f"\nWrote files to {args.out}/")


if __name__ == "__main__":
    main()

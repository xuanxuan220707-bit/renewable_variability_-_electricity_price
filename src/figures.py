"""Figures used by both the notebook and the research report."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

COLORS = ["#177e89", "#ef8354", "#5b5f97", "#71945b", "#c29b40", "#8d6e63", "#c05780"]


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "figure.dpi": 110,
                         "savefig.dpi": 180, "axes.titleweight": "semibold"})


def save(fig, root: Path, name: str) -> Path:
    path = root / "outputs" / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_overview(frame: pd.DataFrame, root: Path) -> Path:
    setup_style()
    observed = frame[frame.observed_physical].copy()
    observed.index = observed.index.tz_convert("Europe/Madrid")
    monthly = observed.resample("MS").mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.0), constrained_layout=True)
    for column, label, color in [("load_mw", "Demand", COLORS[0]), ("wind_mw", "Onshore wind", COLORS[2]),
                                  ("solar_mw", "Solar", COLORS[4]), ("residual_mw", "Residual load", COLORS[1])]:
        axes[0].plot(monthly.index, monthly[column]/1000, label=label, color=color, linewidth=1.5)
    axes[0].set(ylabel="Monthly mean power (GW)", title="Demand and wind-solar supply")
    axes[0].legend(ncol=2, fontsize=8, loc="upper left")
    axes[1].plot(monthly.index, monthly.price_eur_mwh, color=COLORS[0], linewidth=1.8)
    axes[1].set(ylabel="Monthly mean price (EUR/MWh)", title="Reported actual electricity price")
    for ax in axes:
        ax.axvline(pd.Timestamp("2017-01-01", tz="Europe/Madrid"), ls="--", color=".5", lw=.8)
        ax.axvline(pd.Timestamp("2018-01-01", tz="Europe/Madrid"), ls="--", color=".5", lw=.8)
        ax.tick_params(axis="x", rotation=25)
    return save(fig, root, "01_market_overview")


def plot_profiles(frame: pd.DataFrame, root: Path) -> Path:
    test = frame[(frame.year == 2018) & frame.observed_physical]
    hourly = test.groupby("hour").mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.7), constrained_layout=True)
    for column, label, color in [("load_mw", "Demand", COLORS[0]), ("wind_mw", "Onshore wind", COLORS[2]), ("solar_mw", "Solar", COLORS[4])]:
        axes[0].plot(hourly.index, hourly[column]/1000, color=color, label=label)
    axes[0].set(xlabel="Hour in Spain", ylabel="Mean power (GW)", title="Hourly generation and demand, 2018", xticks=[0,6,12,18,23])
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].plot(hourly.index, hourly.price_eur_mwh, color=COLORS[1], marker="o", ms=3)
    axes[1].set(xlabel="Hour in Spain", ylabel="Mean price (EUR/MWh)", title="Hourly price profile, 2018", xticks=[0,6,12,18,23])
    return save(fig, root, "02_hourly_profiles")


def plot_regimes(regimes: pd.DataFrame, root: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.7), constrained_layout=True)
    names = ["Low wind + solar", "High wind + solar"]
    x = np.arange(2)
    for j, load in enumerate(["Low load", "High load"]):
        values = regimes.loc[load].reindex(names)
        for ax, column, factor in [(axes[0], "mean_price", 1), (axes[1], "high_price_rate", 100)]:
            bars = ax.bar(x + (j-.5)*.32, values[column]*factor, width=.32, color=COLORS[j], label=load)
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    for ax in axes:
        ax.set_xticks(x, names)
        ax.legend(fontsize=8, ncol=2, loc="upper right")
    axes[0].set(ylabel="Mean price (EUR/MWh)", title="Load and generation regimes, 2018", ylim=(0,100))
    axes[1].set(ylabel="High-price hours (%)", title="Frequency above the training price threshold", ylim=(0,85))
    return save(fig, root, "03_load_renewable_regimes")


def plot_performance(result: dict, root: Path) -> Path:
    test = result["features"].loc[result["masks"]["test"]]
    y = test.target.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), constrained_layout=True)
    names = ["Calendar logistic", "Price + calendar", "Full logistic", "Full forest", "Full boosting"]
    for i, name in enumerate(names):
        p = result["probabilities"][name]["test"]
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(recall, precision, color=COLORS[i], label=name, lw=1.4)
        observed, predicted = calibration_curve(y, p, n_bins=8, strategy="quantile")
        if name != "Calendar logistic":
            axes[1].plot(predicted, observed, color=COLORS[i], marker="o", ms=3, label=name)
    axes[0].axhline(y.mean(), color=".4", ls="--", label=f"Event rate: {y.mean():.1%}")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Six-hour-ahead ranking, 2018", xlim=(0,1), ylim=(0,1.02))
    axes[0].legend(fontsize=7, loc="lower left")
    axes[1].plot([0,1],[0,1],color=".4",ls="--")
    axes[1].set(xlabel="Mean predicted probability", ylabel="Observed event frequency", title="Reliability by probability bin", xlim=(0,1), ylim=(0,1))
    axes[1].legend(fontsize=7, loc="lower right")
    return save(fig, root, "04_forecast_performance")


def plot_errors(result: dict, root: Path) -> Path:
    test = result["features"].loc[result["masks"]["test"]]
    name = result["selected"]
    p = result["probabilities"][name]["test"]
    y = test.target.to_numpy()
    alarm = p >= result["thresholds"][name]
    matrix = confusion_matrix(y, alarm, labels=[0,1])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.0), constrained_layout=True)
    axes[0].imshow(matrix, cmap="Blues")
    axes[0].grid(False)
    for i in range(2):
        for j in range(2):
            axes[0].text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=14,
                         color="white" if matrix[i,j] > matrix.max()/2 else "black")
    axes[0].set(xticks=[0,1], xticklabels=["No alarm", "Alarm"], yticks=[0,1],
                yticklabels=["Normal price", "High price"], xlabel="Prediction", ylabel="Outcome", title=f"{name}: alarm outcomes")
    month = test.target_time.dt.tz_convert("Europe/Madrid").dt.month.to_numpy()
    monthly = []
    for value in range(1,13):
        mask = month == value
        monthly.append({"Month":value, "AP":average_precision_score(y[mask],p[mask]) if y[mask].sum() else np.nan,
                        "Event_rate":y[mask].mean()})
    table = pd.DataFrame(monthly)
    table.to_csv(root / "outputs" / "monthly_performance.csv", index=False)
    axes[1].plot(table.Month,table.AP,color=COLORS[0],marker="o",label="Average precision")
    axes[1].plot(table.Month,table.Event_rate,color=COLORS[1],marker="s",label="High-price frequency")
    axes[1].set(xlabel="Target month in 2018",ylabel="Score / frequency",title="Monthly ranking and event prevalence",xticks=range(1,13),ylim=(0,1))
    axes[1].legend(fontsize=8)
    return save(fig, root, "05_alarm_errors_monthly")


def plot_importance(importance: pd.DataFrame, root: Path) -> Path:
    values = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.0, 2.7), constrained_layout=True)
    ax.barh(values.Group,values.AP_drop,xerr=values.SD,color=COLORS[0],alpha=.9,capsize=3)
    ax.axvline(0,color=".4",lw=.8)
    ax.set(xlabel="Decrease in test average precision (mean +/- SD)",title="Full boosting: disruption of information groups")
    return save(fig, root, "06_group_importance")


def plot_decisions(curve: pd.DataFrame, decisions: pd.DataFrame, root: Path) -> Path:
    fig, axes = plt.subplots(1,2,figsize=(10.5,2.9),constrained_layout=True)
    axes[0].plot(curve.False_alarms_per_1000,curve.Misses_per_1000,color=COLORS[0])
    axes[0].scatter(decisions.False_alarms_per_1000,decisions.Misses_per_1000,color=COLORS[1],zorder=3)
    for row in decisions.itertuples():
        offset = (5, 5) if row.Miss_cost_ratio != 20 else (7, -14)
        axes[0].annotate(f"r={row.Miss_cost_ratio}",(row.False_alarms_per_1000,row.Misses_per_1000),xytext=offset,textcoords="offset points",fontsize=8)
    axes[0].set(xlabel="False alarms per 1,000 forecasts",ylabel="Missed events per 1,000 forecasts",title="Test alarm trade-off")
    axes[1].plot(decisions.Miss_cost_ratio,100*decisions.Alarm_rate,marker="o",color=COLORS[0],label="Alarm rate")
    axes[1].plot(decisions.Miss_cost_ratio,100*decisions.Recall,marker="s",color=COLORS[1],label="Event recall")
    axes[1].set(xlabel="Relative cost of a missed event, r",ylabel="Percent",title="Policies chosen on validation data",ylim=(0,105))
    axes[1].legend(fontsize=8)
    return save(fig, root, "07_alarm_tradeoffs")


def plot_case(frame: pd.DataFrame, result: dict, root: Path) -> Path:
    test_frame = frame[frame.year == 2018].copy()
    local_index = test_frame.index.tz_convert("Europe/Madrid")
    day = pd.Series(test_frame.high_price.to_numpy(),index=local_index).resample("D").sum()
    # The earliest seven-day window maximizing observed high-price hours is selected for illustration.
    end = day.rolling(7,min_periods=7).sum().idxmax() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=7)
    case = test_frame[(local_index >= start) & (local_index < end)]
    name = result["selected"]
    features = result["features"].loc[result["masks"]["test"]]
    target_index = pd.DatetimeIndex(features.target_time)
    target_local = target_index.tz_convert("Europe/Madrid")
    select = (target_local >= start) & (target_local < end)
    p = result["probabilities"][name]["test"][select]
    fig, axes = plt.subplots(3,1,figsize=(10.5,5.2),sharex=True,constrained_layout=True)
    local_case = case.index.tz_convert("Europe/Madrid")
    axes[0].plot(local_case,case.residual_mw/1000,color=COLORS[0],label="Residual load")
    axes[0].plot(local_case,case.renewable_mw/1000,color=COLORS[2],label="Wind + solar")
    axes[0].set(ylabel="GW",title=f"Illustrative high-price week: {start.date()} to {(end-pd.Timedelta(days=1)).date()}")
    axes[0].legend(fontsize=8,ncol=2,loc="upper right")
    axes[1].plot(local_case,case.price_eur_mwh,color=COLORS[1],label="Actual price")
    threshold = frame.loc[frame.year <= 2016,"price_eur_mwh"].quantile(.9)
    axes[1].axhline(threshold,color=".4",ls="--",label="Training 90th percentile")
    axes[1].set(ylabel="EUR/MWh")
    axes[1].legend(fontsize=8,ncol=2,loc="lower right")
    axes[2].plot(target_local[select],p,color=COLORS[0],label="Six-hour-ahead probability")
    axes[2].plot(local_case,case.risk_index,color=COLORS[2],alpha=.7,label="Contemporaneous risk index")
    axes[2].axhline(result["thresholds"][name],color=COLORS[0],ls="--",lw=.9,label="Validation alarm threshold")
    axes[2].set(ylabel="Probability / index",xlabel="Target time in Spain",ylim=(0,1.03))
    axes[2].legend(fontsize=7,ncol=3,loc="lower right")
    pd.DataFrame({"start_local":[str(start)],"end_local_exclusive":[str(end)],
                  "high_price_hours":[int(case.high_price.sum())],"hours":[len(case)]}).to_csv(root/"outputs"/"case_week.csv",index=False)
    return save(fig, root, "08_warning_case")

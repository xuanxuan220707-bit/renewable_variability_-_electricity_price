from pathlib import Path

import pandas as pd


def describe_data(frame: pd.DataFrame, thresholds: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    physical = frame[frame.observed_physical].copy()
    annual = physical.groupby("year").agg(
        hours=("load_mw", "size"), load_gw=("load_mw", lambda x: x.mean() / 1000),
        wind_gw=("wind_mw", lambda x: x.mean() / 1000), solar_gw=("solar_mw", lambda x: x.mean() / 1000),
        renewable_share=("renewable_share", "mean"), price=("price_eur_mwh", "mean"),
        high_price_rate=("high_price", "mean"), compound_rate=("compound_event", "mean"))
    test = physical[physical.year == 2018].copy()
    test["load_group"] = test.load_mw.gt(thresholds["load_median"]).map({True: "High load", False: "Low load"})
    test["renewable_group"] = test.renewable_mw.gt(thresholds["renewable_median"]).map({True: "High wind + solar", False: "Low wind + solar"})
    regimes = test.groupby(["load_group", "renewable_group"]).agg(
        hours=("price_eur_mwh", "size"), mean_price=("price_eur_mwh", "mean"),
        high_price_rate=("high_price", "mean"), residual_gw=("residual_mw", lambda x: x.mean()/1000))
    annual.to_csv(root / "outputs" / "annual_summary.csv")
    regimes.to_csv(root / "outputs" / "regime_summary.csv")
    correlations = physical[["load_mw", "wind_mw", "solar_mw", "residual_mw", "price_eur_mwh", "temperature_c", "wind_speed_ms"]].corr(method="spearman")
    correlations.to_csv(root / "outputs" / "spearman_correlations.csv")
    return annual, regimes

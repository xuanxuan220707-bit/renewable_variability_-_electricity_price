from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ENERGY = {"total load actual": "load_mw", "generation wind onshore": "wind_mw",
          "generation solar": "solar_mw", "price actual": "price_eur_mwh"}
WEATHER = {"temp": "temperature_c", "wind_speed": "wind_speed_ms",
           "humidity": "humidity_pct", "clouds_all": "cloud_cover_pct", "pressure": "pressure_hpa"}
BOUNDS = {"temp": (180, 340), "wind_speed": (0, 60), "humidity": (0, 100),
          "clouds_all": (0, 100), "pressure": (870, 1085)}
CITIES = {"Madrid", "Barcelona", "Valencia", "Seville", "Bilbao"}


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns {sorted(missing)}")


def validate_and_prepare(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = root / "data" / "raw"
    energy = pd.read_csv(raw / "energy_dataset.csv")
    weather = pd.read_csv(raw / "weather_features.csv")
    require_columns(energy, {"time", *ENERGY}, "Energy")
    require_columns(weather, {"dt_iso", "city_name", *WEATHER}, "Weather")
    energy["timestamp"] = pd.to_datetime(energy["time"], utc=True, errors="raise")
    weather["timestamp"] = pd.to_datetime(weather["dt_iso"], utc=True, errors="raise")
    weather["city_name"] = weather["city_name"].str.strip()
    if set(weather.city_name) != CITIES:
        raise ValueError("Unexpected weather city coverage.")
    if energy.timestamp.duplicated().any():
        raise ValueError("Duplicate electricity timestamps.")
    energy = energy.sort_values("timestamp").set_index("timestamp")
    index = pd.date_range(energy.index.min(), energy.index.max(), freq="h", tz="UTC")
    if not energy.index.equals(index):
        raise ValueError("Electricity timestamps are not a complete hourly grid.")
    for column in ENERGY:
        energy[column] = pd.to_numeric(energy[column], errors="raise")
        if np.isinf(energy[column]).any():
            raise ValueError(f"Infinite value in {column}")
        if column != "price actual" and (energy[column].dropna() < 0).any():
            raise ValueError(f"Negative physical measurement: {column}")
    if energy["price actual"].isna().any():
        raise ValueError("Price targets must be observed.")
    audit = [{"Check": "Electricity rows", "Count": len(energy)},
             {"Check": "Weather rows", "Count": len(weather)},
             {"Check": "Duplicate city-hour rows", "Count": int(weather.duplicated(["timestamp", "city_name"]).sum())}]
    for original, renamed in ENERGY.items():
        audit.append({"Check": f"Missing {renamed}", "Count": int(energy[original].isna().sum())})
    for column, (lower, upper) in BOUNDS.items():
        weather[column] = pd.to_numeric(weather[column], errors="raise")
        invalid = weather[column].notna() & ~weather[column].between(lower, upper)
        audit.append({"Check": f"Out-of-range weather {column}", "Count": int(invalid.sum())})
        weather.loc[invalid, column] = np.nan
    # Equal weighting of cities follows averaging within each city-hour.
    city_hour = weather.groupby(["timestamp", "city_name"])[list(WEATHER)].mean()
    city_coverage = city_hour.reset_index().groupby("timestamp").city_name.nunique().reindex(index)
    filled_cities = []
    for city in sorted(CITIES):
        values = city_hour.xs(city, level="city_name").reindex(index).ffill(limit=6)
        values["city_name"] = city
        filled_cities.append(values)
    averages = pd.concat(filled_cities).groupby(level=0)[list(WEATHER)].mean().rename(columns=WEATHER)
    averages["temperature_c"] -= 273.15
    observed = energy[list(ENERGY)].rename(columns=ENERGY)
    frame = observed.ffill(limit=6).join(averages)
    frame["observed_physical"] = observed[["load_mw", "wind_mw", "solar_mw"]].notna().all(axis=1)
    frame["weather_city_count"] = city_coverage
    frame["renewable_mw"] = frame.wind_mw + frame.solar_mw
    frame["residual_mw"] = frame.load_mw - frame.renewable_mw
    frame["renewable_share"] = frame.renewable_mw / frame.load_mw
    frame["price_change"] = frame.price_eur_mwh.diff().abs()
    local = frame.index.tz_convert("Europe/Madrid")
    frame["year"] = local.year
    frame["month"] = local.month
    frame["hour"] = local.hour
    audit.append({"Check": "Hours with fewer than five observed cities", "Count": int((city_coverage < 5).sum())})
    audit.append({"Check": "Hours excluded from physical summaries", "Count": int((~frame.observed_physical).sum())})
    quality = pd.DataFrame(audit)
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed / "hourly_data.csv", index_label="timestamp")
    quality.to_csv(root / "outputs" / "data_quality.csv", index=False)
    details = {"electricity_start_utc": str(index.min()), "electricity_end_utc": str(index.max()),
               "cities": sorted(CITIES), "weather_bounds": BOUNDS,
               "electricity_missing": {v: int(energy[k].isna().sum()) for k, v in ENERGY.items()}}
    (root / "outputs" / "data_validation.json").write_text(json.dumps(details, indent=2) + "\n")
    return frame, quality, details


def empirical_rank(reference: pd.Series, values: pd.Series) -> np.ndarray:
    sorted_values = np.sort(reference.dropna().to_numpy())
    if not len(sorted_values):
        raise ValueError("Empty rank reference.")
    result = np.searchsorted(sorted_values, values.to_numpy(), side="right") / len(sorted_values)
    result = result.astype(float)
    result[values.isna().to_numpy()] = np.nan
    return result


def add_risk_indicators(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = frame.copy()
    training = frame[frame.year <= 2016]
    physical_training = training[training.observed_physical]
    thresholds = {"price_q90": float(training.price_eur_mwh.quantile(.9)),
                  "load_median": float(physical_training.load_mw.median()),
                  "renewable_median": float(physical_training.renewable_mw.median()),
                  "residual_q75": float(physical_training.residual_mw.quantile(.75))}
    frame["high_price"] = frame.price_eur_mwh > thresholds["price_q90"]
    frame["high_pressure"] = frame.residual_mw > thresholds["residual_q75"]
    frame["compound_event"] = frame.high_price & frame.high_pressure
    for source, result in [("residual_mw", "pressure_rank"), ("price_eur_mwh", "price_rank"),
                           ("price_change", "change_rank")]:
        reference = physical_training[source] if source == "residual_mw" else training[source]
        frame[result] = empirical_rank(reference, frame[source])
    frame["risk_index"] = .45 * frame.pressure_rank + .45 * frame.price_rank + .10 * frame.change_rank
    return frame, thresholds


def build_features(frame: pd.DataFrame, horizon: int = 6) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    x = pd.DataFrame(index=frame.index)
    for column in ["price_eur_mwh", "load_mw", "wind_mw", "solar_mw", "residual_mw", "renewable_share", *WEATHER.values()]:
        x[column] = frame[column]
    for lag in [1, 24, 168]:
        x[f"price_lag_{lag}"] = frame.price_eur_mwh.shift(lag)
    x["price_mean_24"] = frame.price_eur_mwh.rolling(24, min_periods=24).mean()
    x["price_std_24"] = frame.price_eur_mwh.rolling(24, min_periods=24).std()
    x["price_delta_1"] = frame.price_eur_mwh.diff()
    for column in ["load_mw", "wind_mw", "solar_mw", "residual_mw"]:
        x[f"{column}_delta_1"] = frame[column].diff()
        x[f"{column}_lag_24"] = frame[column].shift(24)
    target_local = (frame.index + pd.Timedelta(hours=horizon)).tz_convert("Europe/Madrid")
    for name, values, period in [("hour", target_local.hour, 24), ("weekday", target_local.dayofweek, 7),
                                 ("month", target_local.month - 1, 12)]:
        x[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        x[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    x["target_price"] = frame.price_eur_mwh.shift(-horizon)
    x["target"] = frame.high_price.shift(-horizon).astype("Float64")
    x["target_time"] = frame.index + pd.Timedelta(hours=horizon)
    x["year"] = frame.year
    # Burn-in and missing future targets are removed; other predictor gaps are imputed on training data.
    x = x[x.price_lag_168.notna() & x.target.notna()].copy()
    x["target"] = x.target.astype(int)
    excluded = {"target_price", "target", "target_time", "year"}
    columns = [column for column in x.columns if column not in excluded]
    groups = {
        "Price history": [column for column in columns if column.startswith("price")],
        "Demand": [column for column in columns if column.startswith("load")],
        "Renewables and residual load": [column for column in columns if column.startswith(("wind_mw", "solar_mw", "residual", "renewable"))],
        "Weather": list(WEATHER.values()),
        "Calendar": [column for column in columns if column.endswith(("_sin", "_cos"))],
    }
    assert sorted(sum(groups.values(), [])) == sorted(columns)
    return x, groups


def chronological_masks(features: pd.DataFrame) -> dict[str, pd.Series]:
    target_year = features.target_time.dt.tz_convert("Europe/Madrid").dt.year
    return {"train": (features.year <= 2016) & (target_year <= 2016),
            "validation": (features.year == 2017) & (target_year == 2017),
            "test": (features.year == 2018) & (target_year == 2018)}

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table(headers, rows, alignment):
    lines = ["\\begin{tabular}{" + alignment + "}", "\\toprule",
             " & ".join(headers) + " \\\\", "\\midrule"]
    lines += [" & ".join(map(str, row)) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(root: Path = ROOT):
    output = root / "outputs"
    destination = output / "tables"
    destination.mkdir(exist_ok=True)
    quality = pd.read_csv(output / "data_quality.csv").set_index("Check").Count
    keys = [("Electricity observations", "Electricity rows"), ("Weather observations", "Weather rows"),
            ("Repeated city-hour rows", "Duplicate city-hour rows"), ("Missing demand measurements", "Missing load_mw"),
            ("Missing onshore wind measurements", "Missing wind_mw"), ("Missing solar measurements", "Missing solar_mw"),
            ("Out-of-range wind-speed readings", "Out-of-range weather wind_speed"),
            ("Out-of-range pressure readings", "Out-of-range weather pressure"),
            ("Hours excluded from physical summaries", "Hours excluded from physical summaries")]
    (destination / "data_quality.tex").write_text(table(["Data check", "Count"],
        [[label, f"{int(quality[key]):,}"] for label, key in keys], "lr"))
    annual = pd.read_csv(output / "annual_summary.csv")
    rows = [[int(r.year), f"{int(r.hours):,}", f"{r.load_gw:.2f}", f"{r.wind_gw:.2f}", f"{r.solar_gw:.2f}",
             f"{100*r.renewable_share:.1f}", f"{r.price:.2f}", f"{100*r.high_price_rate:.1f}", f"{100*r.compound_rate:.1f}"]
            for r in annual.itertuples()]
    (destination / "annual_summary.tex").write_text(table(["Year", "Hours", "Load", "Wind", "Solar", "Share", "Price", "High", "Joint"], rows, "lrrrrrrrr"))
    splits = pd.read_csv(output / "split_summary.csv")
    rows = [[r.Split.capitalize(), "2015--2016" if r.Split == "train" else "2017" if r.Split == "validation" else "2018",
             f"{r.Origins:,}", f"{r.Events:,}", f"{100*r.Event_rate:.2f}"] for r in splits.itertuples()]
    (destination / "split_summary.tex").write_text(table(["Partition", "Origin years", "Forecasts", "Events", "Event rate (\\%)"], rows, "llrrr"))
    metrics = pd.read_csv(output / "model_metrics.csv")
    validation = metrics[metrics.Split == "validation"].set_index("Model")
    aliases = {"Calendar logistic": "Calendar", "Price + calendar": "Price + calendar", "Full logistic": "Full logistic",
               "Full forest": "Full forest", "Full boosting": "Full boosting", "Boosting without weather": "No weather",
               "Boosting without renewables": "No wind--solar"}
    rows = [[aliases[r.Model], f"{validation.loc[r.Model, 'AP']:.3f}", f"{r.AP:.3f}", f"{r.ROC_AUC:.3f}",
             f"{r.Brier:.3f}", f"{r.Precision:.3f}", f"{r.Recall:.3f}", f"{r.F1:.3f}", f"{r.Threshold:.2f}"]
            for r in metrics[metrics.Split == "test"].itertuples()]
    (destination / "model_metrics.tex").write_text(table(["Model", "Val. AP", "Test AP", "AUC", "Brier", "Prec.", "Recall", "$F_1$", "$\\tau$"], rows, "lrrrrrrrr"))
    bootstrap = pd.read_csv(output / "bootstrap_comparison.csv")
    rows = [[r.Quantity, f"{r.Estimate:.3f}", f"{r.Lower_95:.3f}", f"{r.Upper_95:.3f}"] for r in bootstrap.itertuples()]
    (destination / "bootstrap_comparison.tex").write_text(table(["Quantity", "Estimate", "2.5th percentile", "97.5th percentile"], rows, "lrrr"))
    decisions = pd.read_csv(output / "decision_scenarios.csv")
    rows = [[r.Miss_cost_ratio, f"{r.Threshold:.2f}", f"{100*r.Alarm_rate:.1f}", f"{100*r.Recall:.1f}",
             f"{r.False_alarms_per_1000:.1f}", f"{r.Misses_per_1000:.1f}", f"{r.Test_relative_loss:.3f}"] for r in decisions.itertuples()]
    (destination / "decision_scenarios.tex").write_text(table(["$r$", "$\\tau$", "Alarms (\\%)", "Recall (\\%)", "FP/1,000", "FN/1,000", "Loss"], rows, "rrrrrrr"))


if __name__ == "__main__":
    main()

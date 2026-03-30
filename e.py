from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path("runs/holdout_run")
EVENTS_CSV = ROOT / "holdout_stage1_tp_fp_events.csv"
XGB_CSV = ROOT / "shap_stage1" / "xgb_tp_fp_shap_long.csv"
TCN_CSV = ROOT / "shap_stage1" / "tcn_gru_tp_fp_shap_long.csv"
LAGS_CSV = ROOT / "shap_stage1" / "tcn_gru_tp_fp_top_lags.csv"

# output folder in the project root, not inside runs/holdout_run
OUTDIR = Path("tp_fp_shap_reports")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_pair(rows: list[dict[str, str]], race_id: int, driver_id: int) -> list[dict[str, str]]:
    out = []
    for r in rows:
        try:
            if int(r["race_id"]) == race_id and int(r["driver_id"]) == driver_id:
                out.append(r)
        except (KeyError, ValueError):
            continue
    return out


def group_by_lap(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[int(r["lapno"])].append(r)
    return dict(sorted(grouped.items()))


def build_report(race_id: int, driver_id: int) -> str:
    event_rows = filter_pair(read_csv(EVENTS_CSV), race_id, driver_id)
    xgb_rows = filter_pair(read_csv(XGB_CSV), race_id, driver_id)
    tcn_rows = filter_pair(read_csv(TCN_CSV), race_id, driver_id)
    lag_rows = filter_pair(read_csv(LAGS_CSV), race_id, driver_id)

    lines: list[str] = []
    lines.append(f"race={race_id}, driver={driver_id}")
    lines.append("")

    lines.append("=== TP/FP event laps ===")
    if not event_rows:
        lines.append("No predicted positive laps found.")
        return "\n".join(lines)

    event_rows = sorted(event_rows, key=lambda r: int(r["lapno"]))
    laps = [int(r["lapno"]) for r in event_rows]
    laps_to_keep = set(laps)

    for r in event_rows:
        lines.append(
            f"lap={int(r['lapno']):<3}  "
            f"event_type={r['event_type']:<2}  "
            f"final_p_pit={float(r['p_pit']):.6f}  "
            f"race_progress_pct={float(r['race_progress_pct']):.2f}"
        )

    lines.append("")
    lines.append("--- Matching XGB SHAP rows ---")
    if not xgb_rows:
        lines.append("No XGB SHAP rows found.")
    else:
        grouped = group_by_lap([r for r in xgb_rows if int(r["lapno"]) in laps_to_keep])
        for lap, lap_rows in grouped.items():
            lines.append("")
            lines.append(f"lap {lap}")
            for r in sorted(lap_rows, key=lambda x: int(x["rank"])):
                lines.append(
                    f"  rank={int(r['rank']):>2}  "
                    f"event={r['event_type']:<2}  "
                    f"feature={r['feature']:<35}  "
                    f"shap={float(r['shap_value']): .6f}  "
                    f"abs={float(r['abs_shap_value']): .6f}  "
                    f"value={r['feature_value']}"
                )

    lines.append("")
    lines.append("--- Matching TCN-GRU SHAP rows ---")
    if not tcn_rows:
        lines.append("No TCN-GRU SHAP rows found.")
    else:
        grouped = group_by_lap([r for r in tcn_rows if int(r["lapno"]) in laps_to_keep])
        for lap, lap_rows in grouped.items():
            lines.append("")
            lines.append(f"lap {lap}")
            for r in sorted(lap_rows, key=lambda x: int(x["rank"])):
                lines.append(
                    f"  rank={int(r['rank']):>2}  "
                    f"event={r['event_type']:<2}  "
                    f"feature={r['feature']:<35}  "
                    f"signed_sum={float(r['signed_shap_sum_over_time']): .6f}  "
                    f"abs_sum={float(r['abs_shap_sum_over_time']): .6f}"
                )

    lines.append("")
    lines.append("--- Top TCN-GRU lags for those laps ---")
    if not lag_rows:
        lines.append("No lag rows found.")
    else:
        grouped = group_by_lap([r for r in lag_rows if int(r["lapno"]) in laps_to_keep])
        for lap, lap_rows in grouped.items():
            lines.append("")
            lines.append(f"lap {lap}")
            for r in sorted(lap_rows, key=lambda x: int(x["rank"])):
                lines.append(
                    f"  rank={int(r['rank']):>2}  "
                    f"lag={int(r['lag']):>2}  "
                    f"abs_shap_at_lag={float(r['abs_shap_at_lag']): .6f}"
                )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write TP/FP SHAP inspection reports for race-driver pairs."
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("RACE_ID", "DRIVER_ID"),
        required=True,
        help="Race-driver pair, e.g. --pair 73 17",
    )
    parser.add_argument(
        "--outdir",
        default=str(OUTDIR),
        help="Output folder in the project root. Default: tp_fp_shap_reports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for race_s, driver_s in args.pair:
        race_id = int(race_s)
        driver_id = int(driver_s)

        report = build_report(race_id, driver_id)
        outpath = outdir / f"race_{race_id}_driver_{driver_id}_tp_fp_shap_report.txt"
        outpath.write_text(report, encoding="utf-8")
        print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
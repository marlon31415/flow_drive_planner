"""
Utility to extract the final closed-loop reactive agents score (plus per-scenario-type
scores) from a nuplan simulation output directory and append them to a persistent CSV file.

Also extracts navigation compliance submetric and nav-adjusted score from a separate
aggregator if available.

A markdown rendering of the CSV is written alongside it as <scores_file>.md, rebuilt from the
full file on every append.

Usage (standalone):
    python extract_score.py <output_dir> <epoch> [--scores-file val_scores.csv] [--cleanup]

Usage (as library):
    from extract_score import extract_and_log_score
    score = extract_and_log_score(output_dir, epoch, scores_file="val_scores.csv", cleanup=True)
"""

import argparse
import glob
import os
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from csv_to_markdown_table import csv_to_markdown, default_output_path

# Component metrics that make up the overall score:
#   Score = weighted-avg(progress, TTC, speed-limit, comfort)
#           x prod(no_at_fault_collision, drivable_area_compliance,
#                   ego_is_making_progress, driving_direction)
# All of them are present as columns in the weighted-average aggregator
# parquet alongside the final score.
SCORE_COMPONENT_METRICS = [
    "ego_progress_along_expert_route",
    "time_to_collision_within_bound",
    "speed_limit_compliance",
    "ego_is_comfortable",
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "ego_is_making_progress",
    "driving_direction_compliance",
]


def find_aggregator_parquet(output_dir: str) -> str:
    """
    Find the closed-loop reactive agents weighted average aggregator parquet file.
    Returns the path to the parquet file.
    """
    pattern = os.path.join(
        output_dir,
        "aggregator_metric",
        "closed_loop_reactive_agents_weighted_average_metrics_*.parquet",
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No aggregator parquet found matching: {pattern}\n"
            f"Contents of output_dir: {os.listdir(output_dir) if os.path.isdir(output_dir) else 'DIR NOT FOUND'}"
        )
    # Take the latest one if multiple exist
    return sorted(matches)[-1]


def extract_scores(parquet_path: str) -> Tuple[float, Dict[str, float]]:
    """
    Read the aggregator parquet and extract:
      - The final overall weighted-average score
      - Per-scenario-type scores

    The parquet contains:
      - Individual scenario rows (scenario_type is set, scenario is the token)
      - Scenario-type aggregate rows (scenario == scenario_type name)
      - A 'final_score' row (overall score)

    Returns: (final_score, {scenario_type: score})
    """
    df = pd.read_parquet(parquet_path)

    # --- Final score ---
    final_rows = df[df["scenario"] == "final_score"]
    if final_rows.empty:
        final_score = float(df.iloc[-1]["score"])
    else:
        final_score = float(final_rows.iloc[0]["score"])

    # --- Per-scenario-type scores ---
    # Scenario-type aggregate rows are those where scenario == scenario_type
    # but not 'final_score'
    scenario_type_scores: Dict[str, float] = OrderedDict()
    type_rows = df[
        (df["scenario"] == df["scenario_type"]) & (df["scenario"] != "final_score")
    ]
    for _, row in type_rows.iterrows():
        st = row["scenario_type"]
        sc = row["score"]
        if st and pd.notna(sc):
            scenario_type_scores[str(st)] = float(sc)

    return final_score, scenario_type_scores


def extract_score_components(parquet_path: str) -> Dict[str, float]:
    """
    Read the aggregator parquet and extract the individual metrics that
    make up the overall score (see SCORE_COMPONENT_METRICS) from the
    'final_score' row.
    """
    df = pd.read_parquet(parquet_path)

    final_rows = df[df["scenario"] == "final_score"]
    row = final_rows.iloc[0] if not final_rows.empty else df.iloc[-1]

    components: Dict[str, float] = OrderedDict()
    for metric in SCORE_COMPONENT_METRICS:
        if metric in df.columns and pd.notna(row.get(metric)):
            components[metric] = float(row[metric])

    return components


def find_nav_aggregator_parquet(output_dir: str) -> Optional[str]:
    """
    Find the navigation-compliance-aware aggregator parquet file.
    Returns the path, or None if not found.
    """
    pattern = os.path.join(
        output_dir,
        "aggregator_metric",
        "closed_loop_reactive_agents_nav_weighted_average_metrics_*.parquet",
    )
    matches = glob.glob(pattern)
    if not matches:
        return None
    return sorted(matches)[-1]


def extract_nav_compliance(
    parquet_path: str,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Extract navigation compliance submetric and nav-adjusted score from the
    nav aggregator parquet.

    Returns:
        (nav_score, nav_compliance, {scenario_type: nav_compliance_value})

    nav_score: final score with navigation compliance as multiplicative factor.
    nav_compliance: overall navigation compliance submetric (fraction of
        on-route scenarios, weighted by scenario count per type).
    per-type nav_compliance: average navigation_compliance per scenario type.
    """
    df = pd.read_parquet(parquet_path)

    # --- Nav-adjusted final score ---
    final_rows = df[df["scenario"] == "final_score"]
    if final_rows.empty:
        nav_score = float(df.iloc[-1]["score"])
    else:
        nav_score = float(final_rows.iloc[0]["score"])

    # --- Navigation compliance submetric ---
    nav_col = "navigation_compliance"
    nav_compliance = 0.0
    per_type_nav: Dict[str, float] = OrderedDict()

    if nav_col in df.columns:
        # Final row already contains the average (sum / total_scenarios) from the aggregator
        if not final_rows.empty and pd.notna(final_rows.iloc[0].get(nav_col)):
            nav_compliance = float(final_rows.iloc[0][nav_col])

        # Per-scenario-type navigation compliance
        type_rows = df[
            (df["scenario"] == df["scenario_type"]) & (df["scenario"] != "final_score")
        ]
        for _, row in type_rows.iterrows():
            st = row["scenario_type"]
            nc = row.get(nav_col)
            num = row.get("num_scenarios", 1)
            if st and pd.notna(nc) and num:
                per_type_nav[str(st)] = float(nc) / float(num)

    return nav_score, nav_compliance, per_type_nav


def append_score_to_csv(
    scores_file: str,
    epoch: int,
    final_score: float,
    scenario_type_scores: Dict[str, float],
    nav_score: Optional[float] = None,
    nav_compliance: Optional[float] = None,
    per_type_nav_compliance: Optional[Dict[str, float]] = None,
    score_components: Optional[Dict[str, float]] = None,
    markdown: bool = True,
) -> None:
    """
    Append a row with epoch, final score, per-scenario-type scores, and timestamp.
    Columns: epoch, score, nav_score, nav_compliance, <score component metrics>,
             <scenario_type_1>, ..., nc_<scenario_type_1>, ..., timestamp
    If the file exists, new scenario-type columns are added as needed.

    Unless ``markdown`` is off, a markdown table of the whole file is (re)written beside it.
    """
    scores_dir = os.path.dirname(scores_file)
    if scores_dir:
        os.makedirs(scores_dir, exist_ok=True)

    # Sorted scenario type keys for deterministic column order
    sorted_types = sorted(scenario_type_scores.keys())

    if os.path.isfile(scores_file):
        # Read existing to merge column sets
        existing_df = pd.read_csv(scores_file)
        existing_cols = list(existing_df.columns)
    else:
        existing_df = None
        existing_cols = []

    if score_components is None:
        score_components = {}

    # Build the full column set
    fixed_cols = [
        "epoch",
        "score",
        "nav_score",
        "nav_compliance",
    ] + SCORE_COMPONENT_METRICS
    # Merge existing scenario-type columns with new ones (preserving order)
    existing_type_cols = [
        c
        for c in existing_cols
        if c not in fixed_cols + ["timestamp"] and not c.startswith("nc_")
    ]
    all_type_cols = list(OrderedDict.fromkeys(existing_type_cols + sorted_types))

    # Navigation compliance per-type columns (nc_<scenario_type>)
    if per_type_nav_compliance is None:
        per_type_nav_compliance = {}
    nc_sorted_types = sorted(per_type_nav_compliance.keys())
    existing_nc_cols = [c for c in existing_cols if c.startswith("nc_")]
    new_nc_cols = [f"nc_{st}" for st in nc_sorted_types]
    all_nc_cols = list(OrderedDict.fromkeys(existing_nc_cols + new_nc_cols))

    all_cols = fixed_cols + all_type_cols + all_nc_cols + ["timestamp"]

    # Build new row (scores as percentages: 0-100 with 2 decimal places)
    new_row = {
        "epoch": epoch,
        "score": f"{final_score * 100:.2f}",
        "nav_score": f"{nav_score * 100:.2f}" if nav_score is not None else "",
        "nav_compliance": (
            f"{nav_compliance * 100:.2f}" if nav_compliance is not None else ""
        ),
        "timestamp": datetime.now().isoformat(),
    }
    for metric in SCORE_COMPONENT_METRICS:
        val = score_components.get(metric, None)
        new_row[metric] = f"{val * 100:.2f}" if val is not None else ""
    for st in all_type_cols:
        val = scenario_type_scores.get(st, None)
        new_row[st] = f"{val * 100:.2f}" if val is not None else ""
    for nc_col in all_nc_cols:
        st = nc_col[3:]  # strip "nc_" prefix
        val = per_type_nav_compliance.get(st, None)
        new_row[nc_col] = f"{val * 100:.2f}" if val is not None else ""

    # If existing data, reindex to match new column set
    if existing_df is not None:
        existing_df = existing_df.reindex(columns=all_cols, fill_value="")
        new_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        new_df = pd.DataFrame([new_row], columns=all_cols)

    new_df.to_csv(scores_file, index=False)

    if markdown:
        # Rendered from the file just written rather than from new_df: read_csv turns the blanks
        # of a column-union row into NaN, which would reach the table as "nan".
        markdown_path = default_output_path(Path(scores_file))
        markdown_path.write_text(csv_to_markdown(Path(scores_file)), encoding="utf-8")

    print(
        f"[extract_score] Epoch {epoch}: score={final_score * 100:.2f} -> {scores_file}"
    )
    if nav_score is not None:
        print(f"  nav_score: {nav_score * 100:.2f}")
    if nav_compliance is not None:
        print(f"  nav_compliance: {nav_compliance * 100:.2f}")
    for metric in SCORE_COMPONENT_METRICS:
        val = score_components.get(metric)
        if val is not None:
            print(f"  {metric}: {val * 100:.2f}")
    for st, sc in scenario_type_scores.items():
        nc_val = per_type_nav_compliance.get(st)
        nc_str = f" (nc: {nc_val * 100:.2f})" if nc_val is not None else ""
        print(f"  {st}: {sc * 100:.2f}{nc_str}")


def extract_and_log_score(
    output_dir: str,
    epoch: int,
    scores_file: str = "val_scores.csv",
    cleanup: bool = True,
    markdown: bool = True,
) -> float:
    """
    End-to-end: find parquet, extract final + per-scenario-type scores,
    append to CSV, optionally clean up.
    Returns the final score.
    """
    parquet_path = find_aggregator_parquet(output_dir)
    final_score, scenario_type_scores = extract_scores(parquet_path)
    score_components = extract_score_components(parquet_path)

    # Extract navigation compliance data if available
    nav_score = None
    nav_compliance_val = None
    per_type_nav = None
    nav_parquet = find_nav_aggregator_parquet(output_dir)
    if nav_parquet is not None:
        nav_score, nav_compliance_val, per_type_nav = extract_nav_compliance(
            nav_parquet
        )

    append_score_to_csv(
        scores_file,
        epoch,
        final_score,
        scenario_type_scores,
        nav_score=nav_score,
        nav_compliance=nav_compliance_val,
        per_type_nav_compliance=per_type_nav,
        score_components=score_components,
        markdown=markdown,
    )

    if cleanup and os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"[extract_score] Cleaned up simulation output: {output_dir}")

    return final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract val14 simulation score")
    parser.add_argument("output_dir", help="Simulation output directory")
    parser.add_argument("epoch", type=int, help="Training epoch number")
    parser.add_argument(
        "--scores-file",
        default="simulation_scores/val_scores.csv",
        help="CSV file to append score to",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the simulation output directory after extracting score",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip the markdown table written alongside the CSV",
    )
    args = parser.parse_args()

    extract_and_log_score(
        args.output_dir,
        args.epoch,
        args.scores_file,
        args.cleanup,
        markdown=not args.no_markdown,
    )

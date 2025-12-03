import csv
import os
from typing import Any, Dict, Optional


def save_timing_to_csv(
    profiler,
    output_path: str,
    model_size: str,
    phase: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Extract timing from profiler.recorded_durations and save to CSV.

    Args:
        profiler: The Analyzer's profiler object with recorded_durations
        output_path: Path to save the CSV file
        model_size: Model size string (e.g., "0.5B", "7B")
        phase: Phase name ("factors" or "scores")
        extra_metadata: Additional columns to include in each row
    """
    if extra_metadata is None:
        extra_metadata = {}

    recorded_durations = profiler.recorded_durations

    if not recorded_durations:
        print(f"No timing data to save for {phase}")
        return

    # Calculate total duration for percentage
    total_duration = 0.0
    for durations in recorded_durations.values():
        total_duration += sum(durations)

    # Prepare rows
    rows = []
    for action, durations in recorded_durations.items():
        num_calls = len(durations)
        total_time = sum(durations)
        mean_time = total_time / num_calls if num_calls > 0 else 0.0
        percentage = (total_time / total_duration * 100) if total_duration > 0 else 0.0

        row = {
            "model_size": model_size,
            "phase": phase,
            "action": action,
            "mean_duration_s": round(mean_time, 6),
            "num_calls": num_calls,
            "total_duration_s": round(total_time, 6),
            "percentage": round(percentage, 2),
        }
        row.update(extra_metadata)
        rows.append(row)

    # Sort by total duration descending
    rows.sort(key=lambda x: x["total_duration_s"], reverse=True)

    # Determine fieldnames
    base_fields = ["model_size", "phase", "action", "mean_duration_s", "num_calls", "total_duration_s", "percentage"]
    extra_fields = list(extra_metadata.keys())
    fieldnames = base_fields + extra_fields

    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(output_path)

    # Write to CSV
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Timing data saved to {output_path}")

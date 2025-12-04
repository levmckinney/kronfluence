"""
Plot timing results from the batch test experiments.
Visualizes time taken for Fit Covariance, Fit Lambda, and Perform Eigendecomposition
across different model sizes and batch sizes.
"""

import matplotlib.pyplot as plt
from typing import List, Literal
import numpy as np
from tueplots import bundles, markers
import pandas as pd
import seaborn as sns
from pathlib import Path
import glob


def init_plotting(nrows: int = 1, ncols: int = 1, column: Literal['half', 'full'] = "half", usetex: bool = True) -> None:
    """Initializes the matplotlib plotting environment with tueplots settings."""
    plt.rcParams.update(
        bundles.icml2024(
            column=column,
            nrows=nrows,
            ncols=ncols,
            usetex=usetex,
        )
    )
    plt.rcParams.update(markers.with_edge())
    plt.rcParams.update({"figure.dpi": 100})


def load_all_results(results_dir: str = "batch_test_results") -> pd.DataFrame:
    """
    Load all timing CSV files from the results directory.

    Args:
        results_dir: Directory containing the timing CSV files

    Returns:
        DataFrame with all timing results combined
    """
    results_path = Path(results_dir)
    csv_pattern = str(results_path / "timing_factors_*.csv")
    csv_files = glob.glob(csv_pattern)

    print(f"Looking for CSV files matching: {csv_pattern}")
    print(f"Found {len(csv_files)} CSV files")

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found matching pattern: {csv_pattern}")

    all_data = []
    for csv_file in csv_files:
        print(f"  Loading: {csv_file}")
        df = pd.read_csv(csv_file)
        all_data.append(df)

    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)

    return combined_df


def prepare_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and prepare data for plotting.
    Focus on the three main phases: Fit Covariance, Fit Lambda, and Perform Eigendecomposition.

    Args:
        df: Combined dataframe with all results

    Returns:
        Filtered dataframe ready for plotting
    """
    # Filter for the three main actions we want to plot
    actions_to_plot = ['Fit Covariance', 'Fit Lambda', 'Perform Eigendecomposition']
    plot_df = df[df['action'].isin(actions_to_plot)].copy()

    # Convert model size to numeric for proper ordering
    # Extract numeric value from model size (e.g., "0.5B" -> 0.5)
    plot_df['model_size_numeric'] = plot_df['model_size'].str.replace('B', '').astype(float)

    # Sort by model size
    plot_df = plot_df.sort_values('model_size_numeric')

    return plot_df


def plot_timing_by_phase(df: pd.DataFrame, output_file: str = "timing_by_phase.pdf"):
    """
    Create a figure with three subplots (columns), one for each phase.
    Each subplot shows time vs model size for different batch sizes.

    Args:
        df: Prepared dataframe with timing data
        output_file: Output filename for the figure
    """
    # Initialize plotting with three columns
    init_plotting(nrows=1, ncols=3, column='full', usetex=False)

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    # Define the three phases to plot
    phases = ['Fit Covariance', 'Fit Lambda', 'Perform Eigendecomposition']

    # Get unique batch sizes and create color palette
    batch_sizes = sorted(df['factor_batch_size'].unique())
    colors = sns.color_palette("husl", n_colors=len(batch_sizes))

    for idx, phase in enumerate(phases):
        ax = axes[idx]

        # Filter data for this phase
        phase_df = df[df['action'] == phase].copy()

        # Plot for each batch size
        for bs_idx, batch_size in enumerate(batch_sizes):
            bs_df = phase_df[phase_df['factor_batch_size'] == batch_size]

            if len(bs_df) > 0:
                # Sort by model size for proper line plotting
                bs_df = bs_df.sort_values('model_size_numeric')

                ax.plot(
                    bs_df['model_size_numeric'],
                    bs_df['mean_duration_s'],
                    marker='o',
                    label=f'Batch Size {batch_size}',
                    color=colors[bs_idx],
                    linewidth=1.5,
                    markersize=6
                )

        # Set labels and title
        ax.set_xlabel('Model Size (B Parameters)')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(phase)

        # Set x-axis to log scale if appropriate
        # ax.set_xscale('log')

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add legend
        if idx == 2:  # Only add legend to the rightmost plot
            ax.legend(loc='upper left', fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")

    # Show figure
    plt.show()


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics of the timing data."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nTotal number of experiments: {len(df)}")
    print(f"Model sizes tested: {sorted(df['model_size'].unique())}")
    print(f"Batch sizes tested: {sorted(df['factor_batch_size'].unique())}")
    print(f"Actions measured: {sorted(df['action'].unique())}")

    print("\n" + "-"*80)
    print("Timing by Action (mean across all experiments):")
    print("-"*80)
    for action in sorted(df['action'].unique()):
        action_df = df[df['action'] == action]
        print(f"{action:30s}: {action_df['mean_duration_s'].mean():10.2f}s (±{action_df['mean_duration_s'].std():.2f}s)")

    print("\n" + "-"*80)
    print("Timing by Model Size (mean across all actions):")
    print("-"*80)
    for model_size in sorted(df['model_size'].unique(), key=lambda x: float(x.replace('B', ''))):
        model_df = df[df['model_size'] == model_size]
        print(f"{model_size:10s}: {model_df['mean_duration_s'].mean():10.2f}s (±{model_df['mean_duration_s'].std():.2f}s)")

    print("\n" + "="*80 + "\n")


def main():
    """Main function to load data and create plots."""
    print("Loading batch test results...")

    # Load all results
    df_all = load_all_results()

    print(f"Loaded {len(df_all)} timing measurements from CSV files")

    # Print summary statistics
    print_summary_statistics(df_all)

    # Show the full dataframe structure
    print("\nFirst few rows of combined data:")
    print(df_all.head(10))

    print("\nDataFrame Info:")
    print(df_all.info())

    # Prepare data for plotting (filter to main phases)
    plot_df = prepare_plot_data(df_all)

    print(f"\nFiltered to {len(plot_df)} measurements for main phases")
    print(f"Phases included: {plot_df['action'].unique()}")

    # Create the timing by phase plot
    print("\nCreating timing by phase plot...")
    plot_timing_by_phase(plot_df, output_file="timing_by_phase.pdf")

    print("\nPlotting complete!")


if __name__ == "__main__":
    main()

import textwrap
import warnings
import pytest
import numpy as np
from scipy.stats import chi2_contingency, chi2
import shutil
import os


def fail_with_message(message: str):
    """Cleanly formatted pytest failure message."""
    pytest.fail(textwrap.dedent(message))

def warn_with_message(message: str):
    """Cleanly formatted warning message."""
    warnings.warn(textwrap.dedent(message), UserWarning)
def format_sigfigs(x, sigfigs=4):
    if x == 0:
        return "0"
    else:
        from math import log10, floor
        digits = sigfigs - 1 - floor(log10(abs(x)))
        return f"{x:.{digits}f}"

def calculate_chi_squared_and_cramers(df, what, alpha=0.05):
    """
    Expects a dataframe with three columns:
    - identifier : any name
    - counts (simulated) : name must be 'count_simulated'
    - counts (historic): name must be 'count_historic'

    what = what is being tested? e.g. callsign, callsign group


    """

    # Create a table of proportions (separate from the original count table)
    df['prop_simulated'] = df['count_simulated'] / df['count_simulated'].sum()
    df['prop_historic'] = df['count_historic'] / df['count_historic'].sum()
    df['abs_diff'] = (df['prop_simulated'] - df['prop_historic']).abs()

    df.round(5).to_csv(f"tests/test_outputs/TEST_OUTPUT_{what}_proportions.csv")

    # Create the contingency table (observed frequencies)
    # Rows: Callsigns, Columns: Data Source (Simulated, Historic)
    contingency_table = df[['count_simulated', 'count_historic']].values

    # Perform the Chi-Squared Test of Homogeneity
    chi2_stat, p_val, dof, expected_freq = chi2_contingency(contingency_table)

    # Interpretation
    alpha = alpha # Significance level

    if p_val < alpha:
        print(f"Result: Reject the null hypothesis (H₀) for {what}.")
        print("Conclusion: There is a statistically significant difference in the distribution of callsigns between the simulated and historic data.")
    else:
        print(f"Result: Fail to reject the null hypothesis (H₀) for {what}.")
        print("Conclusion: There is no statistically significant difference in the distribution of callsigns between the simulated and historic data.")

    # --- Calculate Effect Size (Cramér's V) ---
    n = np.sum(contingency_table) # Total number of observations
    min_dim = min(contingency_table.shape) - 1 # min(rows-1, cols-1)

    # Handle potential division by zero if min_dim is 0 (though unlikely here)
    if min_dim == 0:
        cramers_v = np.nan
        fail_with_message("Cannot calculate Cramér's V because min(rows-1, cols-1) is 0.")
    else:
        cramers_v = np.sqrt(chi2_stat / (n * min_dim))
        print(f"Total Observations (n): {n}")
        print(f"Cramér's V (Effect Size): {cramers_v:.4f}")

        # Interpretation of Cramér's V
        if p_val > alpha:
            if cramers_v > 0.4:
                warn_with_message(
                    f"Some difference observed in {what} proportions, though not statistically significant. "
                    f"p_val: {p_val:.4f}, cramers_v: {cramers_v:.4f}\n\n{df}"
                )
            # else: no action (i.e. p > alpha and low V)
        elif cramers_v < 0.2:
            warn_with_message(
                f"Some difference observed in {what} proportions - be aware. "
                f"p_val: {p_val:.4f}, cramers_v: {cramers_v:.4f}\n\n{df}"
            )
        elif cramers_v < 0.3:
            warn_with_message(
                f"Moderate difference observed in {what} proportions - investigation recommended. "
                f"p_val: {p_val:.4f}, cramers_v: {cramers_v:.4f}\n\n{df}"
            )
        else:
            fail_with_message(
                f"Major difference observed in {what} proportions. "
                f"p_val: {p_val:.4f}, cramers_v: {cramers_v:.4f}\n\n{df}"
            )

    max_diff = df['abs_diff'].max()

    # Additionally assert that no single category differs too much
    if max_diff > 0.1:
        fail_with_message(f"{what}: One category differs in proportion by more than 10% (found {max_diff:.4f}).\n\n{df}")
    elif max_diff >= 0.05:
        warn_with_message(f"{what}: One category differs in proportion by more than 5% (found {max_diff:.4f}).\n\n{df}")
    else:
        pass



def save_logs(output_path, log_location="log.txt", output_folder="tests/test_logs/"):
    full_output_path = f"{output_folder}/LOG_{output_path}"

    if os.path.exists(full_output_path):
            try:
                os.remove(full_output_path)
                print(f"Removed previous log: {full_output_path}")
            except Exception as e:
                print(f"Warning: Failed to remove {full_output_path} — {e}")

    shutil.copyfile(log_location, full_output_path)

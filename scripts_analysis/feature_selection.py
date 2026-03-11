import tabulate
import numpy as np
import pandas as pd


def filter_reliable_feature():
    """
    This method filters out the robust features based on multiple criteria:
    1. Selection Frequency: Only features selected in 100% of the bootstrap samples are considered.
    2. Sign Consistency: Only features with consistent sign across all samples are considered.
    3. Statistical Significance: Only features with p-value ≤ 0.05 are considered.
    4. Effect Size: Only features with a minimum ±5% change in odds are considered.
    """
    # === Load data ===
    df = pd.read_csv(
        "results_feature_selection/robust_results_feature_summary.csv"
    )
    print(tabulate.tabulate(df.head(), headers='keys', tablefmt='psql'))
    print("length of df:", len(df))
    print(df.p_approx.value_counts())

    # === Thresholds ===
    freq_thresh = 1.0
    sign_thresh = 1.0
    p_thresh = 0.00
    odds_change_thresh = 0.05  # <-- minimum ±5% change in odds (adjustable)

    fs = df.copy()

    # === Compute odds ratio and percent change in odds ===
    fs["Odds_Ratio"] = np.exp(fs["Avg_Std_Coef"])
    fs["Percent_Change_Odds"] = (fs["Odds_Ratio"] - 1.0) * 100

    # === Reliable if meets all thresholds ===
    fs["Reliable"] = (
            (fs.Selection_Freq >= freq_thresh)
            & (fs.Sign_Consistency >= sign_thresh)
            & (fs.p_approx <= p_thresh)
            & ~((fs.CI_Lower < 0) & (fs.CI_Upper > 0))  # CI doesn’t cross zero
            & (np.abs(fs["Percent_Change_Odds"]) >= odds_change_thresh * 100)  # min 5% change in odds
    )

    # === Composite reliability score (optional) ===
    fs["Reliability_Score"] = (
            0.2 * fs.Selection_Freq
            + 0.3 * fs.Sign_Consistency
            + 0.2 * (1 - fs.p_approx)
            + 0.3 * np.abs(fs.Avg_Std_Coef)
    )

    fs = fs.sort_values("Reliability_Score", ascending=False)

    # === Diagnostics ===
    print("Number of reliable features:", fs["Reliable"].sum())

    # === Filter only reliable ===
    fs = fs[fs["Reliable"]]

    return fs


if __name__ == '__main__':
    reliable_features = filter_reliable_feature()
    print("Final reliable features:")
    print(tabulate.tabulate(reliable_features.head(), headers='keys', tablefmt='psql'))
    print("Total reliable features:", len(reliable_features))

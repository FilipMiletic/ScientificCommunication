- 145 initial features
- script: elastic net + stability selection ( 5 fold innter, 10 fold out CV), the resulting table has 145 features
with information (file: robust_results_feature_summary.csv)
- selectionf frequency, effect direction, odds, apprx sign.
--> 45 final features (a table with this?) (reliable_features_odds_with_feature_groups.csv)
- rank by absolute change in odds, select top features per feature group --> 24 final features
- unbiased estimates on these 24 features for each dataset: odds_realistic.csv, odds_synthetic.csv


import argparse
import numpy as np
import pandas as pd
import polars as pl
import tabulate
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from scipy.stats import loguniform


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def compute_log_likelihood(y_true, y_pred_proba):
    """Compute log-likelihood from predicted probabilities."""
    eps = 1e-15  # avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    return np.sum(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))


def pseudo_r2_measures(y_true, y_pred_proba):
    """Return both McFadden and Nagelkerke pseudo-R²."""
    n = len(y_true)
    ll_full = compute_log_likelihood(y_true, y_pred_proba)
    ll_null = compute_log_likelihood(y_true, np.repeat(np.mean(y_true), n))
    r2_mcfadden = 1 - (ll_full / ll_null)
    r2_nagelkerke = r2_mcfadden / (1 - ll_null / n)
    return r2_mcfadden, r2_nagelkerke


def get_clean_feature_df(feature_df, features):
    """
    Clean and preprocess the dataframe:
    - Drop text columns and IDs if present
    - Remove columns with arrays/lists
    - Remove columns with all NaNs or one unique value
    - Replace inf values, fill NaNs (numeric -> mean, bool/int -> 0)
    """
    # 1. Drop clearly irrelevant columns
    columns_to_ignore = {"abstract", "full_text", "title", "text", "acl_id", "after"}
    df = feature_df.drop(columns=columns_to_ignore.intersection(feature_df.columns), errors="ignore")
    # only keep features that are in the provided features list
    df = df[[col for col in df.columns if col in features]]

    # 2. Remove columns containing lists/arrays in any row
    def is_array_column(series):
        return series.apply(lambda x: isinstance(x, (list, np.ndarray))).any()

    array_cols = [col for col in df.columns if is_array_column(df[col])]
    df = df.drop(columns=array_cols, errors="ignore")

    # 3. Drop columns with only NaNs or a single unique value
    drop_cols = [col for col in df.columns if df[col].isnull().all() or df[col].nunique() <= 1]
    df = df.drop(columns=drop_cols, errors="ignore")
    print(f"Dropped {len(array_cols) + len(drop_cols)} columns (arrays/NaNs/constants). Remaining shape: {df.shape}")

    # 4. Replace inf/-inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # how many NaNs are in the dataframe now?
    print(f"Dataframe shape after cleaning: {df.shape}. Total NaNs: {df.isna().sum().sum()}")

    # 5. Fill remaining NaNs: numeric -> mean, others -> 0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(0, inplace=True)

    # 6. Final safety check
    df.fillna(0, inplace=True)

    return df




def robust_feature_selection(
    X, y,
    N_OUTER=10, N_INNER=5,
    SEED=42, n_iter_search=20,
    n_bootstrap=200, ci_alpha=0.05,
    save_prefix="robust_results"
):
    """
    Robust feature selection with nested CV Elastic Net Logistic Regression.
    Includes:
      - Averaged standardized coefficients
      - Feature selection frequency & sign consistency
      - Bootstrapped confidence intervals & p-values
      - Saves all outputs to CSV
    """

    # set random seed to ensure reproducibility
    np.random.seed(SEED)
    features = X.columns.tolist()

    # --- Containers to store feature information---
    feature_counts = pd.Series(0, index=features)
    coef_sums = pd.Series(0.0, index=features)
    sign_consistency = pd.Series(0, index=features)
    per_fold_results = []

    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
    # define parameter distribution for RandomizedSearchCV
    param_dist = {
        "logreg__C": loguniform(1e-4, 1),
        "logreg__l1_ratio": np.linspace(0.7, 1.0, 6)
    }

    # === Outer Loop ===
    for fold, (train_idx, test_idx) in enumerate(
        tqdm(outer_cv.split(X, y), total=N_OUTER, desc="Outer CV")
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # scale features and fit Elastic Net Logistic Regression with RandomizedSearchCV
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                penalty="elasticnet", solver="saga",
                max_iter=3000, tol=1e-4, random_state=SEED
            ))
        ])
        # define the search object with the parameter distribution and inner CV, using AUC as the scoring metric
        search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist,
            n_iter=n_iter_search, cv=inner_cv, scoring="roc_auc",
            random_state=SEED, n_jobs=-1, verbose=0
        )
        # fit the model and get the best estimator
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        # extract coefficients and compute AUC on the test set
        coefs = best_model.named_steps["logreg"].coef_.flatten()
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        # Track results
        # 1) only consider features with non-zero coefficients as selected
        selected = coefs != 0
        # 2) if a feature was selected, update its count for that split
        feature_counts[selected] += 1
        # add the standardized coefficient to the sum for that feature (for averaging later)
        coef_sums[selected] += coefs[selected]
        # add the sign of the coefficient to the sign consistency tracker (for averaging later)
        sign_consistency += np.sign(coefs)
        # store per-fold result.
        per_fold_results.append({
            "Fold": fold + 1,
            "AUC": auc,
            "Best_C": search.best_params_["logreg__C"],
            "Best_L1_Ratio": search.best_params_["logreg__l1_ratio"]
        })

    # === Aggregate results ===
    selection_freq = feature_counts / N_OUTER
    # average the coefficients, we want features with consistently large coefficients across folds.
    avg_coefs = coef_sums / feature_counts.replace(0, np.nan)
    # average the sign consistency. We do not want features that flip signs across folds.
    sign_consistency = np.abs(sign_consistency / N_OUTER)
    # we sort the features by selection frequency.
    feature_summary = (
        pd.DataFrame({
            "Selection_Freq": selection_freq,
            "Avg_Std_Coef": avg_coefs.fillna(0),
            "Sign_Consistency": sign_consistency
        })
        .sort_values("Selection_Freq", ascending=False)
    )

    results_df = pd.DataFrame(per_fold_results)
    results_df.to_csv(f"{save_prefix}_per_fold_results.csv", index=False)

    # === Bootstrap for CIs & p-values ===
    print("\nBootstrapping coefficient confidence intervals...")
    from joblib import Parallel, delayed
    # now we still need an idea of the significance of the features. we estimate bootstrapped confidence intervals and p-values for the coefficients.
    def fit_bootstrap(seed):
        np.random.seed(seed)
        idx = np.random.choice(len(y), len(y), replace=True)
        Xb, yb = X.iloc[idx], y.iloc[idx]
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=0.1, l1_ratio=0.9,
                max_iter=3000,
                random_state=seed))
        ])
        pipe.fit(Xb, yb)
        return pipe.named_steps['logreg'].coef_.flatten()

    coef_boot = np.array(Parallel(n_jobs=-1)(
        delayed(fit_bootstrap)(SEED + i) for i in range(n_bootstrap)
    ))

    ci_lower = np.percentile(coef_boot, 100 * ci_alpha / 2, axis=0)
    ci_upper = np.percentile(coef_boot, 100 * (1 - ci_alpha / 2), axis=0)
    p_values = 2 * np.minimum(
        np.mean(coef_boot > 0, axis=0),
        np.mean(coef_boot < 0, axis=0)
    )
    # we add this information to the feature summary dataframe.
    feature_summary["CI_Lower"] = ci_lower
    feature_summary["CI_Upper"] = ci_upper
    feature_summary["p_approx"] = p_values

    # === Save outputs ===
    feature_summary.to_csv(f"{save_prefix}_feature_summary.csv", index=True)
    results_df.to_csv(f"{save_prefix}_nestedcv_summary.csv", index=False)

    print("\n=== Top Stable Features ===")
    print(feature_summary.head(15))

    print(f"\nSaved results to:\n- {save_prefix}_feature_summary.csv\n- {save_prefix}_nestedcv_summary.csv\n- {save_prefix}_per_fold_results.csv")

    return feature_summary, results_df



def find_highly_correlated_features(df1, df2, drop_cols=None, threshold=0.9):
    """Identify sets of highly correlated features in two dataframes."""

    def combine_sets(pairs):
        sets = []
        for a, b in pairs:
            found = False
            for s in sets:
                if a in s or b in s:
                    s.update([a, b])
                    found = True
                    break
            if not found:
                sets.append({a, b})
        return sets

    def get_high_corr(df):
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
        corr_matrix = df.corr().abs()
        high_corr = np.where(corr_matrix > threshold)
        pairs = [
            (corr_matrix.columns[x], corr_matrix.columns[y])
            for x, y in zip(*high_corr)
            if x != y and x < y
        ]
        return combine_sets(pairs)

    c1 = get_high_corr(df1)
    c2 = get_high_corr(df2)

    print("Highly correlated features in df1:", c1)
    print("Highly correlated features in df2:", c2)
    return c1, c2


def main(args):
    df1 = pl.read_parquet(args.input_t1).to_pandas()
    df2 = pl.read_parquet(args.input_t2).to_pandas()
    # our initial set of features (after correlation analysis these are the ones we will keep for the robust feature selection)
    features = ['n_VERB_VerbForm_Fin',
                'n_VERB_VerbForm_Inf',
                'n_VERB_VerbForm_Part',
                'n_VERB_Mood_Ind',
                'n_VERB_Number_Sing',
                'n_NOUN_Number_Plur',
                'n_PRON_PronType_Art',
                'n_PRON_PronType_Dem',
                'n_PRON_PronType_Ind',
                'n_PRON_PronType_Rel',
                'n_PRON_Reflex_Yes',
                'n_PRON_Gender_Fem',
                'n_PRON_Gender_Masc',
                'n_PRON_Gender_Neut',
                'n_ADJ_Degree_Cmp',
                'n_ADJ_Degree_Sup',
                'n_DET_Number_Plur',
                'n_DET_Number_Sing',
                'n_DET_Definite_Def',
                'n_DET_Definite_Ind',
                'n_NUM_Number_Sing',
                'n_ADV_Degree_Cmp',
                'n_ADV_Degree_Pos',
                'n_ADV_Degree_Sup',
                'n_PROPN_Number_Plur',
                'n_PUNCT_PunctType_Brck',
                'n_PUNCT_PunctType_Comm',
                'n_PUNCT_PunctType_Dash',
                'n_PUNCT_PunctType_Peri',
                'n_PUNCT_PunctType_Quot',
                'n_CCONJ_ConjType_Cmp',
                'n_dependency_acl',
                'n_dependency_acomp',
                'n_dependency_advcl',
                'n_dependency_agent',
                'n_dependency_attr',
                'n_dependency_aux',
                'n_dependency_case',
                'n_dependency_compound',
                'n_dependency_conj',
                'n_dependency_csubj',
                'n_dependency_csubjpass',
                'n_dependency_dative',
                'n_dependency_dep',
                'n_dependency_dobj',
                'n_dependency_expl',
                'n_dependency_intj',
                'n_dependency_meta',
                'n_dependency_neg',
                'n_dependency_nsubj',
                'n_dependency_nsubjpass',
                'n_dependency_nummod',
                'n_dependency_oprd',
                'n_dependency_parataxis',
                'n_dependency_pcomp',
                'n_dependency_poss',
                'n_dependency_preconj',
                'n_dependency_prt',
                'n_dependency_quantmod',
                'n_dependency_relcl',
                'n_dependency_xcomp',
                'n_high_Auditory_sensorimotor',
                'n_high_Gustatory_sensorimotor',
                'n_high_Haptic_sensorimotor',
                'n_high_Interoceptive_sensorimotor',
                'n_high_Olfactory_sensorimotor',
                'n_high_Visual_sensorimotor',
                'n_high_Foot_leg_sensorimotor',
                'n_high_Hand_arm_sensorimotor',
                'n_high_Head_sensorimotor',
                'n_high_Mouth_sensorimotor',
                'n_high_Torso_sensorimotor',
                'avg_word_length',
                't_stopword',
                'gunning_fog',
                'compressibility',
                'avg_aoa',
                'n_high_arousal',
                'n_negative_sentiment',
                'n_positive_sentiment',
                'n_high_intensity_anger',
                'n_high_intensity_anticipation',
                'n_high_intensity_disgust',
                'n_high_intensity_joy',
                'n_high_intensity_sadness',
                'n_high_intensity_surprise',
                'n_high_intensity_trust',
                'avg_concreteness',
                'n_high_concreteness',
                'avg_prevalence',
                'tree_width',
                'tree_depth',
                'tree_branching',
                'cttr',
                'simpsons_d',
                'entropy',
                'mtld',
                'mattr',
                'n_global_lemma_hapax_dislegomena',
                'corr_adj_var',
                'corr_adp_var',
                'corr_aux_var',
                'corr_cconj_var',
                'corr_det_var',
                'corr_intj_var',
                'corr_noun_var',
                'corr_num_var',
                'corr_part_var',
                'corr_pron_var',
                'corr_propn_var',
                'corr_punct_var',
                'corr_sconj_var',
                'corr_sym_var',
                'corr_verb_var',
                'corr_space_var',
                'n_adj',
                'n_adv',
                'n_aux',
                'n_intj',
                'n_noun',
                'n_num',
                'n_part',
                'n_pron',
                'n_propn',
                'n_sconj',
                'n_sym',
                'n_x',
                'n_hedges',
                'n_org',
                'n_cardinal',
                'n_date',
                'n_gpe',
                'n_person',
                'n_money',
                'n_product',
                'n_time',
                'n_percent',
                'n_quantity',
                'n_norp',
                'n_loc',
                'n_event',
                'n_ordinal',
                'n_fac',
                'n_law',
                'n_language']

    print("size of T1 dataframe: ", df1.shape)
    print("size of T2 dataframe: ", df2.shape)
    print("type of T1 dataframe: ", type(df1))
    print("type of T2 dataframe: ", type(df2))

    df1["after"] = 0
    df2["after"] = 1

    feature_df = pd.concat([df1, df2], axis=0)
    # print(tabulate(feature_df.head(), headers='keys', tablefmt='psql'))
    print("target distribution:")
    print(feature_df.after.value_counts())

    target = feature_df["after"]
    feature_df = feature_df.drop(columns=["after"], errors="ignore")
    # only select features that are in the predefined features list
    feature_df = feature_df[[col for col in feature_df.columns if col in features]]
    # feature_df = get_clean_feature_df(feature_df, features)

    # only keep columns that have "dependency" in their name
    #    feature_df = feature_df[[col for col in feature_df.columns if "dependency" in col]]
    print(f"Final feature set size: {feature_df.shape[1]} features.")
    # how many NaNs per column?
    print("NaN counts per column:")
    r = feature_df.isna().sum()
    # check which columns have NaNs
    print(r[r > 0])
    # print(tabulate.tabulate(r[r > 0], headers=['Column', 'NaN Count'], tablefmt='psql'))

    print("value counts of the target variable:")
    print(target.value_counts())
    columns_to_ignore = {"abstract", "full_text", "title", "text", "acl_id", "after"}
    feature_df = feature_df.drop(columns=columns_to_ignore.intersection(feature_df.columns), errors="ignore")

    # 2. Remove columns containing lists/arrays in any row
    def is_array_column(series):
        return series.apply(lambda x: isinstance(x, (list, np.ndarray))).any()

    array_cols = [col for col in feature_df.columns if is_array_column(feature_df[col])]
    feature_df = feature_df.drop(columns=array_cols, errors="ignore")
    print(f"Dropped {len(array_cols)} array columns. Remaining shape: {feature_df.shape}")
    # check if all columns have the same length
    lengths = feature_df.apply(len)
    print("Column lengths:")
    print(lengths.value_counts())

    robust_feature_selection(feature_df, target, N_OUTER=args.outer, N_INNER=args.inner, SEED=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust feature selection with nested CV.")
    parser.add_argument("--input_t1", type=str, required=True, help="Path to T1 parquet file")
    parser.add_argument("--input_t2", type=str, required=True, help="Path to T2 parquet file")
    parser.add_argument("--outer", type=int, default=5, help="Number of outer CV folds")
    parser.add_argument("--inner", type=int, default=5, help="Number of inner CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(args)

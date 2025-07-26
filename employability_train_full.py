# employability_train_full.py
# =============================================================================
# Full pipeline WITH heavy EDA, plots, learning/validation curves, etc.
# =============================================================================

import argparse
import warnings
warnings.filterwarnings("ignore")

import math
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.stats import chi2_contingency

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    learning_curve,
    validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE


def savefig(path, tight=True, dpi=150):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def pie_chart_var(col_name, data):
    counts = [g.shape[0] for _, g in data.groupby(col_name)]
    headers = [g.loc[:, col_name].iloc[0] for _, g in data.groupby(col_name)]
    percentages = np.array(counts) / np.array(counts).sum() * 100
    headers = [f"{label} - {round(percentage, 2)}%" for label, percentage in zip(headers, percentages)]
    return counts, headers


def plot_learning_curve_(estimator, title, X, y, out_path):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(.1, 1.0, 5)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation score")
    plt.legend(loc="best")
    savefig(out_path)


def main():
    parser = argparse.ArgumentParser(description="Employability Prediction - FULL (with EDA & plots)")
    parser.add_argument("--data", type=str, default="Student_Employability_dataset_2025.xlsx",
                        help="Path to dataset (.xlsx)")
    parser.add_argument("--outdir", type=str, default="outputs_full",
                        help="Directory to save artifacts/plots/results")
    parser.add_argument("--target", type=str, default="CLASS",
                        help="Target column name (Employable/LessEmployable)")
    parser.add_argument("--drop-cols", nargs="*", default=["Name_of_Student"],
                        help="Columns to drop if present")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    data = pd.read_excel(args.data)
    print("Data loaded:", data.shape)
    print(data.head(10))

    # -------------------------------------------------------------------------
    # Basic Info
    # -------------------------------------------------------------------------
    with open(outdir / "data_info.txt", "w") as f:
        f.write(str(data.info()))
    data.describe(include="all").to_csv(outdir / "describe_all.csv")

    # -------------------------------------------------------------------------
    # Descriptive statistics for numeric columns
    # -------------------------------------------------------------------------
    numeric_data = data.select_dtypes(include='number')
    summary_stats = pd.DataFrame({
        'Count': numeric_data.count(),
        'Mean': numeric_data.mean(),
        'Median': numeric_data.median(),
        'Mode': numeric_data.mode().iloc[0],
        'Std Dev': numeric_data.std(),
        'Min': numeric_data.min(),
        '25%': numeric_data.quantile(0.25),
        '50%': numeric_data.quantile(0.50),
        '75%': numeric_data.quantile(0.75),
        'Max': numeric_data.max(),
        'Skewness': numeric_data.skew()
    })
    summary_stats.to_csv(outdir / "summary_stats.csv")

    # -------------------------------------------------------------------------
    # Univariate Analysis (Histograms + KDE)
    # -------------------------------------------------------------------------
    rf1 = data.select_dtypes(include=['float64', 'int64'])
    sns.set_palette("muted", color_codes=True)
    for col in rf1.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=rf1, x=col, kde=True)
        col_min, col_max = rf1[col].min(), rf1[col].max()
        col_mean, col_median = rf1[col].mean(), rf1[col].median()
        col_mode = rf1[col].mode().iloc[0]
        col_skewness, col_std = rf1[col].skew(), rf1[col].std()
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(
            f"Univariate Analysis of {col}\n"
            f"min={col_min:.2f}, max={col_max:.2f}, "
            f"mean={col_mean:.2f}, median={col_median:.2f}, "
            f"mode={col_mode:.2f}, skewness={col_skewness:.2f}, std={col_std:.2f}"
        )
        savefig(plots_dir / f"univariate_{col}.png")

    # -------------------------------------------------------------------------
    # Bivariate (Chi-square, boxplots)
    # -------------------------------------------------------------------------
    dependent_var = args.target
    data_numeric = data.select_dtypes(include='number')
    chi2_data = []
    boxplot_vars = []

    if dependent_var in data_numeric.columns:
        for col in data_numeric.columns:
            if col != dependent_var:
                ctab = pd.crosstab(data_numeric[col], data_numeric[dependent_var])
                try:
                    chi2_stat, p_val, dof, _ = chi2_contingency(ctab)
                    chi2_data.append({
                        "Variable": col,
                        "Chi-Square": chi2_stat,
                        "p-value": p_val,
                        "Degrees of Freedom": dof
                    })
                    boxplot_vars.append(col)
                except Exception as e:
                    print(f"Chi-square failed for {col}: {e}")

        chi2_df_full = pd.DataFrame(chi2_data).sort_values("p-value")
        chi2_df_full.to_csv(outdir / "chi2_results.csv", index=False)

        for var in boxplot_vars:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data_numeric[dependent_var], y=data_numeric[var], palette='pastel')
            plt.title(f'Boxplot of {var} by {dependent_var}')
            plt.xlabel(dependent_var)
            plt.ylabel(var)
            savefig(plots_dir / f"box_{var}_by_{dependent_var}.png")

    # -------------------------------------------------------------------------
    # Correlation heatmap (Pearson on numeric)
    # -------------------------------------------------------------------------
    numeric_columns = data.select_dtypes(include=['number'])
    if not numeric_columns.empty:
        corr = numeric_columns.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix (Pearson)")
        savefig(plots_dir / "corr_heatmap.png")

    # -------------------------------------------------------------------------
    # Duplicates, nulls, zero-variance, outliers
    # -------------------------------------------------------------------------
    dup_count = data.duplicated().sum()
    nulls = data.isnull().sum()
    zero_var_cols = [col for col in data.columns if data[col].nunique() <= 1]
    with open(outdir / "data_quality.txt", "w") as f:
        f.write(f"Duplicates: {dup_count}\n\n")
        f.write("Nulls:\n")
        f.write(str(nulls))
        f.write("\n\nZero variance columns:\n")
        f.write(str(zero_var_cols))

    # Outlier summary (IQR + Likert manual 1..5)
    outlier_summary = {}
    data_numeric2 = data.select_dtypes(include=['float64', 'int64'])
    for col in data_numeric2.columns:
        Q1 = data_numeric2[col].quantile(0.25)
        Q3 = data_numeric2[col].quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = data_numeric2[(data_numeric2[col] < Q1 - 1.5 * IQR) |
                                     (data_numeric2[col] > Q3 + 1.5 * IQR)]
        manual_outliers = data_numeric2[(data_numeric2[col] < 1) |
                                        (data_numeric2[col] > 5)]
        outlier_summary[col] = {
            "IQR Outlier Count": len(iqr_outliers),
            "Manual Bound Violation Count": len(manual_outliers)
        }
    pd.DataFrame(outlier_summary).T.to_csv(outdir / "outliers_summary.csv")

    # -------------------------------------------------------------------------
    # Drop irrelevant cols + NA
    # -------------------------------------------------------------------------
    data.drop(columns=args.drop_cols, errors='ignore', inplace=True)
    data.dropna(inplace=True)

    # -------------------------------------------------------------------------
    # Encode target
    # -------------------------------------------------------------------------
    if data[args.target].dtype == object:
        data[args.target] = data[args.target].map({'Employable': 1, 'LessEmployable': 0})

    # -------------------------------------------------------------------------
    # Population distribution bar / pie charts
    # -------------------------------------------------------------------------
    counts = data[args.target].value_counts().tolist()
    labels = ['Employable' if val == 1 else 'Less Employable' for val in data[args.target].value_counts().index]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts)
    plt.title("Population Distribution by Employability Class")
    plt.xlabel("Employability Category")
    plt.ylabel("Number of Students")
    savefig(plots_dir / "class_distribution.png")

    # Feature-level pie charts
    col = 3
    row = math.ceil(len(data.columns) / col)
    fig, ax = plt.subplots(row, col, figsize=(20, 10), layout='constrained')
    for idx, ax_ in enumerate(ax.flatten()):
        if idx < len(data.columns):
            col_name = list(data.columns)[idx]
            try:
                counts_, headers_ = pie_chart_var(col_name, data)
                ax_.pie(counts_, labels=headers_)
                ax_.set_title(f"Population {col_name.title()}")
            except Exception:
                ax_.remove()
        else:
            ax_.remove()
    fig.suptitle("Population Pie Chart by Features", fontsize=20)
    savefig(plots_dir / "population_pies.png")

    # -------------------------------------------------------------------------
    # Spearman correlation & pairplot
    # -------------------------------------------------------------------------
    if not data.select_dtypes(include='number').empty:
        spearman_corr = data.corr(method='spearman')
        plt.figure(figsize=(12, 10))
        sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.title("Spearman Rank Correlation Matrix")
        savefig(plots_dir / "spearman_corr.png")

        correlation_with_target = spearman_corr[args.target].drop(args.target).sort_values(ascending=False)
        correlation_with_target.to_csv(outdir / "spearman_corr_with_target.csv")

        # list top 3 correlation per feature
        with open(outdir / "spearman_top3.txt", "w") as f:
            for col in spearman_corr.columns:
                top3 = spearman_corr[col].sort_values(ascending=False).index[:4]
                f.write(f"{top3[0]} is correlated to the student's: {list(top3[1:])}\n")

    try:
        plot = sns.pairplot(data, kind='reg', hue=args.target, height=1.8)
        plot.savefig(plots_dir / "pairplot.png", dpi=120)
        plt.close()
    except Exception as e:
        print("Pairplot failed:", e)

    # -------------------------------------------------------------------------
    # Normalize numeric features (MinMax)
    # -------------------------------------------------------------------------
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    scaler_minmax = MinMaxScaler()
    data[num_cols] = scaler_minmax.fit_transform(data[num_cols])

    # -------------------------------------------------------------------------
    # Train/test, SMOTE, scaling
    # -------------------------------------------------------------------------
    X = data.drop([args.target], axis=1)
    y = data[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler_std = StandardScaler()
    X_train_scaled = scaler_std.fit_transform(X_train_res)
    X_test_scaled = scaler_std.transform(X_test)

    # -------------------------------------------------------------------------
    # Models + GridSearch
    # -------------------------------------------------------------------------
    model_defs = {
        'SVM': (SVC(probability=True, random_state=42), {
            'C': [0.1, 1, 10],
            'gamma': [0.1, 1, 10],
            'kernel': ['rbf']
        }),
        'Decision Tree': (DecisionTreeClassifier(random_state=42), {
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }),
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), {
            'C': [0.1, 1, 10],
            'penalty': ['l2']
        }),
        'KNN': (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        })
    }

    mpl_styles = mpl.style.available
    try:
        mpl.style.use('seaborn-v0_8-deep' if 'seaborn-v0_8-deep' in mpl_styles else mpl_styles[0])
    except Exception:
        mpl.style.use('ggplot')

    results = {}
    best_model = None
    best_f1 = 0

    for name, (model, params) in model_defs.items():
        print(f"Training {name}...")
        grid = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train_scaled, y_train_res)

        best_est = grid.best_estimator_
        y_pred = best_est.predict(X_test_scaled)
        y_prob = best_est.predict_proba(X_test_scaled)[:, 1]

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'best_params': grid.best_params_,
            'report': classification_report(y_test, y_pred, output_dict=False)
        }

        if results[name]['f1'] > best_f1:
            best_f1 = results[name]['f1']
            best_model = best_est

    # Save the best model + scalers
    joblib.dump(best_model, outdir / 'employability_predictor.pkl')
    joblib.dump(scaler_std, outdir / 'scaler.pkl')
    joblib.dump(scaler_minmax, outdir / 'minmax_scaler.pkl')

    # Save results
    results_df = pd.DataFrame({
        k: {m: (v[m] if m != "confusion_matrix" else None) for m in v}  # exclude cm in the csv
        for k, v in results.items()
    }).T
    results_df.to_csv(outdir / 'model_performance.csv')
    print(results_df)

    # Save confusion matrices as plots
    plt.figure(figsize=(12, 8))
    for i, (name, res) in enumerate(results.items(), 1):
        plt.subplot(2, 2, i)
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Less Employable', 'Employable'],
                    yticklabels=['Less Employable', 'Employable'])
        plt.title(f"{name} (F1 = {res['f1']:.3f})")
    savefig(plots_dir / 'confusion_matrices.png')

    # Write text reports
    with open(outdir / "classification_reports.txt", "w") as f:
        for name, res in results.items():
            f.write(f"=== {name} ===\n")
            f.write(res["report"])
            f.write("\n\n")

    # Learning curve for the best SVM if present
    if "SVM" in results:
        best_svm_params = results["SVM"]["best_params"]
        best_svm_model = SVC(probability=True, random_state=42, **best_svm_params)
        plot_learning_curve_(best_svm_model, "Learning Curve: SVM",
                             X_train_scaled, y_train_res,
                             plots_dir / "learning_curve_svm.png")

        # Validation curve on C for SVM (gamma fixed from best params if exists)
        gamma = best_svm_params.get("gamma", 0.1)
        param_range = [0.01, 0.1, 1, 10, 100]
        train_scores, test_scores = validation_curve(
            SVC(kernel='rbf', gamma=gamma, probability=True),
            X_train_scaled, y_train_res,
            param_name='C', param_range=param_range,
            cv=5, scoring='f1', n_jobs=-1
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure()
        plt.semilogx(param_range, train_scores_mean, label="Training F1 Score", marker='o')
        plt.semilogx(param_range, test_scores_mean, label="CV F1 Score", marker='s')
        plt.title(f"Validation Curve (SVM, gamma={gamma})")
        plt.xlabel("C (log scale)")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)
        savefig(plots_dir / "validation_curve_svm.png")

    # Final sanity check predictions
    y_pred = best_model.predict(X_test_scaled)
    with open(outdir / "final_test_predictions.txt", "w") as f:
        f.write("Counts:\n")
        f.write(str(pd.Series(y_pred).value_counts()))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))

    print("All done. Outputs saved to:", outdir.resolve())


if __name__ == "__main__":
    main()

import math
from pathlib import Path
from typing import Dict, List, Tuple

import scipy.signal as scipy_signal

if not hasattr(scipy_signal, "gaussian"):
    from scipy.signal.windows import gaussian as scipy_signal_gaussian

    scipy_signal.gaussian = scipy_signal_gaussian

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR.parent / "output" / "gemini_eval" / "results_breakdowns.csv"
OUTPUT_DIR = BASE_DIR / "output"
METRIC = "ndcg@10"

CATEGORICAL_COLS = [
    "embedding_task_setup",
    "doc_text_variant",
    "model",
    "normalize_embeddings",
    "output_dimensionality",
    "query_text_variant",
    "similarity_metric",
]


def load_and_filter_data(path: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["grouping"] == "overall"].copy()
    df = df[df[metric].notnull()].copy()
    return df


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, List[str]]]:
    encoded: Dict[str, np.ndarray] = {}
    levels_by_col: Dict[str, List[str]] = {}
    df = df.copy()
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        codes, levels = pd.factorize(df[col], sort=True)
        encoded[col] = codes
        levels_by_col[col] = list(levels)
    return df, encoded, levels_by_col


def build_coords(df: pd.DataFrame, levels_by_col: Dict[str, List[str]]) -> Dict[str, List[str]]:
    coords: Dict[str, List[str]] = {"obs": np.arange(len(df))}
    for col, levels in levels_by_col.items():
        coords[f"{col}_level"] = levels
    return coords


def build_model(
    df: pd.DataFrame,
    encoded: Dict[str, np.ndarray],
    levels_by_col: Dict[str, List[str]],
    metric: str,
) -> pm.Model:
    y = df[metric].to_numpy(dtype=float)
    coords = build_coords(df, levels_by_col)

    with pm.Model(coords=coords) as model:
        # Normal likelihood is used first for interpretability.
        # A Beta likelihood or logit-normal likelihood could be explored later.
        y_obs = pm.MutableData("y_obs", y, dims="obs")

        index_data = {}
        for col in CATEGORICAL_COLS:
            index_data[col] = pm.MutableData(f"{col}_idx", encoded[col], dims="obs")

        def centered_effect(name: str, level_dim: str):
            tau = pm.HalfNormal(f"tau_{name}", sigma=0.05)
            z = pm.Normal(f"z_{name}", mu=0, sigma=1, dims=level_dim)
            raw = pm.Deterministic(f"raw_{name}", z * tau, dims=level_dim)
            centered = pm.Deterministic(
                f"effect_{name}",
                raw - pm.math.mean(raw),
                dims=level_dim,
            )
            return centered

        intercept = pm.Normal("intercept", mu=y.mean(), sigma=0.1)
        sigma = pm.HalfNormal("sigma", sigma=0.02)

        effects = {}
        for col in CATEGORICAL_COLS:
            effects[col] = centered_effect(col, f"{col}_level")

        mu = intercept
        for col in CATEGORICAL_COLS:
            mu = mu + effects[col][index_data[col]]

        pm.Deterministic("mu", mu, dims="obs")
        pm.Normal("metric", mu=mu, sigma=sigma, observed=y_obs, dims="obs")

    return model


def export_model_graph(model: pm.Model, output_dir: Path) -> None:
    try:
        graph = pm.model_to_graphviz(model)
    except ImportError as exc:
        print(f"Skipping model graph export: {exc}")
        return

    dot_path = output_dir / "model_graph.dot"
    png_stem = output_dir / "model_graph"

    graph.save(str(dot_path))
    try:
        graph.render(str(png_stem), format="png", cleanup=True)
    except Exception as exc:
        print(f"Saved {dot_path}, but PNG render failed: {exc}")


def sample_model(model: pm.Model) -> az.InferenceData:
    with model:
        idata = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.98,
            random_seed=42,
        )
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=42))
    return idata


def stacked_draws(data_array) -> np.ndarray:
    stacked = data_array.stack(sample=("chain", "draw")).transpose(..., "sample")
    return np.asarray(stacked)


def hdi_bounds(samples: np.ndarray, prob: float) -> Tuple[np.ndarray, np.ndarray]:
    hdi = az.hdi(samples.T, hdi_prob=prob)
    return np.asarray(hdi[:, 0]), np.asarray(hdi[:, 1])


def summarize_effects(
    idata: az.InferenceData,
    levels_by_col: Dict[str, List[str]],
) -> pd.DataFrame:
    rows = []
    for predictor, levels in levels_by_col.items():
        var_name = f"effect_{predictor}"
        effect_samples = stacked_draws(idata.posterior[var_name])
        means = effect_samples.mean(axis=1)
        low80, high80 = hdi_bounds(effect_samples, 0.80)
        low95, high95 = hdi_bounds(effect_samples, 0.95)
        prob_positive = (effect_samples > 0).mean(axis=1)
        for idx, level in enumerate(levels):
            rows.append(
                {
                    "predictor": predictor,
                    "level": level,
                    "posterior_mean_effect": means[idx],
                    "hdi_80_low": low80[idx],
                    "hdi_80_high": high80[idx],
                    "hdi_95_low": low95[idx],
                    "hdi_95_high": high95[idx],
                    "prob_effect_positive": prob_positive[idx],
                }
            )
    return pd.DataFrame(rows)


def summarize_configs(
    idata: az.InferenceData,
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    mu_samples = stacked_draws(idata.posterior["mu"])
    posterior_mean = mu_samples.mean(axis=1)
    low80, high80 = hdi_bounds(mu_samples, 0.80)
    low95, high95 = hdi_bounds(mu_samples, 0.95)

    best_indices = np.argmax(mu_samples.T, axis=1)
    prob_best = np.bincount(best_indices, minlength=len(df)) / mu_samples.shape[1]

    result = df[
        [
            "run_id",
            "embedding_task_setup",
            "doc_text_variant",
            "model",
            "normalize_embeddings",
            "output_dimensionality",
            "query_text_variant",
            "similarity_metric",
        ]
    ].copy()
    result["obs_index"] = np.arange(len(df))
    result["metric"] = metric
    result["observed_value"] = df[metric].to_numpy()
    result["posterior_mean"] = posterior_mean
    result["hdi_80_low"] = low80
    result["hdi_80_high"] = high80
    result["hdi_95_low"] = low95
    result["hdi_95_high"] = high95
    result["prob_best"] = prob_best
    result = result.sort_values("posterior_mean", ascending=False).reset_index(drop=True)
    return result


def save_diagnostics(idata: az.InferenceData, output_dir: Path) -> None:
    print(az.summary(idata, var_names=["intercept", "sigma", "tau_"], filter_vars="like"))

    fig = az.plot_trace(idata, var_names=["intercept", "sigma"])
    plt.savefig(output_dir / "trace_intercept_sigma.png", bbox_inches="tight")
    plt.close()

    fig = az.plot_rank(idata, var_names=["intercept", "sigma"])
    plt.savefig(output_dir / "rank_intercept_sigma.png", bbox_inches="tight")
    plt.close()


def plot_intercept_posterior(idata: az.InferenceData, output_dir: Path) -> None:
    samples = stacked_draws(idata.posterior["intercept"]).reshape(-1)
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=40, color="steelblue", alpha=0.8)
    plt.axvline(samples.mean(), color="black", linestyle="--", linewidth=1)
    plt.title("Posterior of Global Intercept")
    plt.xlabel("Intercept")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "intercept_posterior.png")
    plt.close()


def plot_tau_importance(idata: az.InferenceData, output_dir: Path) -> None:
    rows = []
    for predictor in CATEGORICAL_COLS:
        tau_samples = stacked_draws(idata.posterior[f"tau_{predictor}"]).reshape(-1)
        rows.append(
            {
                "predictor": predictor,
                "mean": tau_samples.mean(),
                "low95": np.quantile(tau_samples, 0.025),
                "high95": np.quantile(tau_samples, 0.975),
            }
        )
    tau_df = pd.DataFrame(rows).sort_values("mean", ascending=True)

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(tau_df))
    plt.errorbar(
        tau_df["mean"],
        y_pos,
        xerr=[
            tau_df["mean"] - tau_df["low95"],
            tau_df["high95"] - tau_df["mean"],
        ],
        fmt="o",
        color="darkred",
        ecolor="gray",
        capsize=4,
    )
    plt.yticks(y_pos, tau_df["predictor"])
    plt.xlabel("Posterior mean tau with 95% interval")
    plt.title("Hyperparameter Family Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "tau_importance.png")
    plt.close()


def plot_effects(effect_summary: pd.DataFrame, output_dir: Path) -> None:
    for predictor in sorted(effect_summary["predictor"].unique()):
        subset = effect_summary[effect_summary["predictor"] == predictor].copy()
        subset = subset.sort_values("posterior_mean_effect")
        y_pos = np.arange(len(subset))

        plt.figure(figsize=(10, max(4, 0.45 * len(subset))))
        plt.errorbar(
            subset["posterior_mean_effect"],
            y_pos,
            xerr=[
                subset["posterior_mean_effect"] - subset["hdi_95_low"],
                subset["hdi_95_high"] - subset["posterior_mean_effect"],
            ],
            fmt="o",
            color="navy",
            ecolor="lightgray",
            capsize=4,
            label="95% CI",
        )
        plt.errorbar(
            subset["posterior_mean_effect"],
            y_pos,
            xerr=[
                subset["posterior_mean_effect"] - subset["hdi_80_low"],
                subset["hdi_80_high"] - subset["posterior_mean_effect"],
            ],
            fmt="o",
            color="navy",
            ecolor="steelblue",
            capsize=6,
            label="80% CI",
        )
        plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
        plt.yticks(y_pos, subset["level"])
        plt.xlabel("Posterior effect relative to grand mean")
        plt.title(f"Effects for {predictor}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"effects_{predictor}.png")
        plt.close()


def plot_observed_vs_predicted(config_summary: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(config_summary["observed_value"], config_summary["posterior_mean"], alpha=0.8)
    min_val = min(config_summary["observed_value"].min(), config_summary["posterior_mean"].min())
    max_val = max(config_summary["observed_value"].max(), config_summary["posterior_mean"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black")
    plt.xlabel("Observed metric")
    plt.ylabel("Posterior mean predicted metric")
    plt.title("Observed vs Posterior Mean Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "observed_vs_predicted.png")
    plt.close()


def plot_residuals(config_summary: pd.DataFrame, output_dir: Path) -> None:
    residuals = config_summary["observed_value"] - config_summary["posterior_mean"]
    plt.figure(figsize=(8, 5))
    plt.scatter(config_summary["posterior_mean"], residuals, alpha=0.8)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Posterior mean predicted metric")
    plt.ylabel("Residual (observed - predicted)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals.png")
    plt.close()


def plot_prob_best(config_summary: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    top = config_summary.sort_values("prob_best", ascending=False).head(top_n).copy()
    short_labels = [f"C{i}" for i in range(1, len(top) + 1)]
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.bar(np.arange(len(top)), top["prob_best"], color="darkgreen")
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(short_labels)
    ax.set_ylabel("Posterior probability best")
    ax.set_title("Top Configurations by Probability of Being Best")
    mapping_lines = []
    for index, row in enumerate(top.itertuples(), start=1):
        mapping_lines.append(
            (
                f"C{index}: model={row.model}, setup={row.embedding_task_setup}, "
                f"dim={row.output_dimensionality}, sim={row.similarity_metric}, "
                f"doc={row.doc_text_variant}, query={row.query_text_variant}, "
                f"norm={row.normalize_embeddings}"
            )
        )
    mapping_text = "\n".join(mapping_lines)
    fig.text(
        0.02,
        0.02,
        mapping_text,
        fontsize=8,
        family="monospace",
        va="bottom",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.42)
    fig.savefig(output_dir / "top_config_prob_best.png")
    plt.close(fig)


def plot_top_config_posteriors(
    idata: az.InferenceData,
    config_summary: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> None:
    mu_samples = stacked_draws(idata.posterior["mu"])
    top = config_summary.head(top_n).copy()

    fig, axes = plt.subplots(top_n, 1, figsize=(12, max(2.2 * top_n, 6)), sharex=True)
    if top_n == 1:
        axes = [axes]

    for ax, row in zip(axes, top.itertuples()):
        samples = mu_samples[row.obs_index, :]
        ax.hist(samples, bins=40, color="steelblue", alpha=0.8, density=True)
        ax.axvline(samples.mean(), color="black", linestyle="--", linewidth=1)
        ax.set_ylabel(row.run_id[:8], rotation=0, labelpad=28, va="center")
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_title(
            f"{row.model} | {row.embedding_task_setup} | dim={row.output_dimensionality} | "
            f"sim={row.similarity_metric} | norm={row.normalize_embeddings}",
            fontsize=10,
        )

    for ax in axes:
        ax.set_xlabel("Posterior predicted metric")
    fig.suptitle("Posterior Predicted Performance for Top Configurations", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_dir / "top_config_posteriors.png")
    plt.close(fig)


def make_visualizations(
    idata: az.InferenceData,
    effect_summary: pd.DataFrame,
    config_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_intercept_posterior(idata, output_dir)
    plot_tau_importance(idata, output_dir)
    plot_effects(effect_summary, output_dir)
    plot_observed_vs_predicted(config_summary, output_dir)
    plot_residuals(config_summary, output_dir)
    plot_prob_best(config_summary, output_dir)
    plot_top_config_posteriors(idata, config_summary, output_dir)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_filter_data(INPUT_CSV, METRIC)
    df, encoded, levels_by_col = encode_categoricals(df, CATEGORICAL_COLS)

    model = build_model(df, encoded, levels_by_col, METRIC)
    export_model_graph(model, OUTPUT_DIR)
    idata = sample_model(model)

    save_diagnostics(idata, OUTPUT_DIR)

    effect_summary = summarize_effects(idata, levels_by_col)
    config_summary = summarize_configs(idata, df, METRIC)

    effect_summary.to_csv(OUTPUT_DIR / "bayesian_effect_summary.csv", index=False)
    config_summary.to_csv(OUTPUT_DIR / "bayesian_config_ranking.csv", index=False)

    make_visualizations(idata, effect_summary, config_summary, OUTPUT_DIR)

    print("Interpretation notes:")
    print("- Positive effect means the level tends to increase the metric relative to the grand mean, after averaging over other hyperparameters.")
    print("- A high tau means that hyperparameter family explains more variation.")
    print("- prob_best is not the same as highest observed score; it accounts for uncertainty and shrinkage.")
    print("- Since this model uses only main effects, strong interactions may appear as residual structure.")
    print("- Natural next interactions: doc_text_variant x query_text_variant, model x output_dimensionality, normalize_embeddings x similarity_metric.")


if __name__ == "__main__":
    main()

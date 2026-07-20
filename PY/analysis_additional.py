"""Reanalysis of the Steinmetz et al. (2019) data for the revised manuscript.

The script compares choice and stimulus decoding from regional firing-rate
features and trial-wise inter-area correlation features. It also performs the
controls requested during review: difficulty-stratified decoding, correct/error
analyses, equal-contrast decoding, correct-to-error stimulus generalization,
and pre-movement evaluation.

Usage
-----
Place ``steinmetz_part0.npz``, ``steinmetz_part1.npz``, and
``steinmetz_part2.npz`` in ``./data_repo`` and run
``python analysis_additional.py`` from the parent directory. Results, plots,
and the complete machine-readable output are written to ``./analysis_output``.

Key safeguards relative to the original scripts:
* standardization is fitted inside each cross-validation training fold;
* the two feature sets have the same dimensionality in every session;
* inter-area results are averaged across reproducible random matched-size
  subsets of region pairs rather than an arbitrary first-ten-pairs choice;
* inference uses mouse-level summaries (10 mice), with exact sign-flip tests;
* local and inter-area results are compared with paired tests.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Use an Arial/Calibri-family sans serif throughout the manuscript figures.
# MANUSCRIPT_FONT_FILE may point to a licensed Arial or Calibri TTF file.
_font_file = os.environ.get("MANUSCRIPT_FONT_FILE")
if _font_file and Path(_font_file).is_file():
    font_manager.fontManager.addfont(_font_file)
    _font_name = font_manager.FontProperties(fname=_font_file).get_name()
    plt.rcParams["font.family"] = _font_name
else:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial", "Calibri", "Carlito", "Liberation Sans", "DejaVu Sans"
    ]


DATA_DIR = Path("data_repo")
OUT_DIR = Path("analysis_output")
OUT_DIR.mkdir(exist_ok=True)

WINDOWS = {
    "pre_early": (0, 25),
    "pre_late": (25, 50),
    "post_early": (50, 75),
    "post_late": (75, 100),
}

MIN_NEURONS = 10
MIN_TRIALS = 30
N_PAIR_SUBSETS = 10
RANDOM_SEED = 20260720

warnings.filterwarnings("ignore", category=FutureWarning)


def load_data() -> list[dict]:
    sessions: list[dict] = []
    for path in sorted(DATA_DIR.glob("steinmetz_part*.npz")):
        sessions.extend(np.load(path, allow_pickle=True)["dat"].tolist())
    return sessions


def pipeline() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=RANDOM_SEED,
        ),
    )


def folds_for(y: np.ndarray) -> StratifiedKFold | None:
    counts = np.bincount(y.astype(int), minlength=2)
    n_splits = min(5, int(counts.min()))
    if len(y) < MIN_TRIALS or n_splits < 3:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)


def oof_predictions(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    cv = folds_for(y)
    if cv is None or X.shape[1] == 0 or not np.isfinite(X).all():
        return None
    return cross_val_predict(
        pipeline(), X, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]


def bac(y: np.ndarray, prob: np.ndarray) -> float:
    if len(y) < 4 or len(np.unique(y)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y, prob >= 0.5))


def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between corresponding rows, with zero for constants."""
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt(np.sum(ac * ac, axis=1) * np.sum(bc * bc, axis=1))
    out = np.zeros(a.shape[0], dtype=float)
    valid = den > 0
    out[valid] = np.sum(ac[valid] * bc[valid], axis=1) / den[valid]
    return out


def feature_matrices(
    session: dict, window: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None:
    spks = session["spks"]
    areas = np.asarray(session["brain_area"])
    good = sorted(
        area for area in np.unique(areas) if np.sum(areas == area) >= MIN_NEURONS
    )
    if len(good) < 3:
        return None

    t0, t1 = window
    pop = {
        area: spks[areas == area, :, t0:t1].mean(axis=0).astype(float)
        for area in good
    }
    local = np.column_stack([pop[area].mean(axis=1) for area in good])

    pair_features = []
    for i, area_a in enumerate(good):
        for area_b in good[i + 1 :]:
            pair_features.append(corr_rows(pop[area_a], pop[area_b]))
    inter = np.column_stack(pair_features)

    # Match feature dimensionality: R local regions versus R inter-area pairs.
    d = local.shape[1]
    if inter.shape[1] < d:
        return None
    rng = np.random.default_rng(
        RANDOM_SEED + sum((i + 1) * ord(c) for i, c in enumerate(session["mouse_name"]))
        + int(session["date_exp"].replace("-", ""))
    )
    subsets = [rng.choice(inter.shape[1], d, replace=False) for _ in range(N_PAIR_SUBSETS)]
    return local, inter, subsets


def repeated_scores(
    X: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    subsets: list[np.ndarray] | None = None,
) -> dict[str, float]:
    """OOF balanced accuracy overall and in prespecified trial strata."""
    scores: dict[str, list[float]] = {name: [] for name in masks}
    selections = subsets if subsets is not None else [np.arange(X.shape[1])]
    for sel in selections:
        probs = oof_predictions(X[:, sel], y)
        if probs is None:
            continue
        for name, mask in masks.items():
            value = bac(y[mask], probs[mask])
            if np.isfinite(value):
                scores[name].append(value)
    return {
        name: float(np.mean(values)) if values else float("nan")
        for name, values in scores.items()
    }


def train_correct_test_error(
    X: np.ndarray,
    stimulus: np.ndarray,
    correct: np.ndarray,
    error: np.ndarray,
    subsets: list[np.ndarray] | None = None,
) -> float:
    """Train stimulus decoder on correct trials and test it on error trials."""
    train = correct & (stimulus != 0)
    test = error & (stimulus != 0)
    y_train = (stimulus[train] > 0).astype(int)
    y_test = (stimulus[test] > 0).astype(int)
    if len(y_train) < MIN_TRIALS or len(y_test) < 8:
        return float("nan")
    if np.bincount(y_train, minlength=2).min() < 5 or np.bincount(y_test, minlength=2).min() < 3:
        return float("nan")

    values = []
    selections = subsets if subsets is not None else [np.arange(X.shape[1])]
    for sel in selections:
        model = pipeline()
        model.fit(X[train][:, sel], y_train)
        prob = model.predict_proba(X[test][:, sel])[:, 1]
        values.append(bac(y_test, prob))
    return float(np.mean(values))


def aggregate_by_mouse(rows: list[dict], outcome: str, feature: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["outcome"] == outcome and row["feature"] == feature and np.isfinite(row["value"]):
            grouped[row["mouse"]].append(row["value"])
    return {mouse: float(np.mean(values)) for mouse, values in grouped.items()}


def exact_signflip(values: np.ndarray, null: float = 0.0) -> float:
    values = np.asarray(values, float) - null
    n = len(values)
    if n == 0:
        return float("nan")
    observed = abs(values.mean())
    signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    perm = np.abs((signs * values).mean(axis=1))
    return float(np.mean(perm >= observed - 1e-15))


def bootstrap_ci(values: np.ndarray, n_boot: int = 20000) -> tuple[float, float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(RANDOM_SEED)
    means = np.mean(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def holm(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def add_holm_to_contrasts(items: list[dict]) -> list[dict]:
    """Attach family-wise Holm-adjusted p values to paired contrast records."""
    adjusted = holm([item["signflip_p"] for item in items]) if items else []
    for item, value in zip(items, adjusted):
        item["holm_p"] = value
    return items


def summarize(rows: list[dict], outcomes: list[str]) -> list[dict]:
    summary = []
    paired_p = []
    local_chance_p = []
    inter_chance_p = []
    for outcome in outcomes:
        local = aggregate_by_mouse(rows, outcome, "local")
        inter = aggregate_by_mouse(rows, outcome, "inter")
        mice = sorted(set(local) & set(inter))
        lv = np.array([local[m] for m in mice])
        iv = np.array([inter[m] for m in mice])
        if len(mice) == 0:
            continue
        diff = lv - iv
        lci = bootstrap_ci(lv)
        ici = bootstrap_ci(iv)
        dci = bootstrap_ci(diff)
        p = exact_signflip(diff)
        paired_p.append(p)
        local_p = exact_signflip(lv, 0.5)
        inter_p = exact_signflip(iv, 0.5)
        local_chance_p.append(local_p)
        inter_chance_p.append(inter_p)
        summary.append(
            {
                "outcome": outcome,
                "n_mice": len(mice),
                "local_mean": float(lv.mean()),
                "local_ci": lci,
                "inter_mean": float(iv.mean()),
                "inter_ci": ici,
                "difference_local_minus_inter": float(diff.mean()),
                "difference_ci": dci,
                "paired_signflip_p": p,
                "local_vs_chance_p": local_p,
                "inter_vs_chance_p": inter_p,
            }
        )
    paired_adjusted = holm(paired_p) if paired_p else []
    local_adjusted = holm(local_chance_p) if local_chance_p else []
    inter_adjusted = holm(inter_chance_p) if inter_chance_p else []
    for item, pp, lp, ip in zip(summary, paired_adjusted, local_adjusted, inter_adjusted):
        item["paired_holm_p"] = pp
        item["local_vs_chance_holm_p"] = lp
        item["inter_vs_chance_holm_p"] = ip
    return summary


def paired_contrast(
    rows: list[dict], outcome_a: str, outcome_b: str, feature: str
) -> dict:
    a = aggregate_by_mouse(rows, outcome_a, feature)
    b = aggregate_by_mouse(rows, outcome_b, feature)
    mice = sorted(set(a) & set(b))
    values = np.array([a[m] - b[m] for m in mice])
    return {
        "contrast": f"{outcome_a} minus {outcome_b}",
        "feature": feature,
        "n_mice": len(mice),
        "mean_difference": float(values.mean()),
        "difference_ci": bootstrap_ci(values),
        "signflip_p": exact_signflip(values),
    }


def interaction_contrast(rows: list[dict], post_window: str) -> dict:
    post_l = aggregate_by_mouse(rows, f"choice_{post_window}_all", "local")
    post_i = aggregate_by_mouse(rows, f"choice_{post_window}_all", "inter")
    pre_l = aggregate_by_mouse(rows, "choice_pre_late_all", "local")
    pre_i = aggregate_by_mouse(rows, "choice_pre_late_all", "inter")
    mice = sorted(set(post_l) & set(post_i) & set(pre_l) & set(pre_i))
    values = np.array(
        [(post_l[m] - post_i[m]) - (pre_l[m] - pre_i[m]) for m in mice]
    )
    return {
        "contrast": f"(local - inter) in {post_window} minus (local - inter) in pre_late",
        "n_mice": len(mice),
        "mean_difference": float(values.mean()),
        "difference_ci": bootstrap_ci(values),
        "signflip_p": exact_signflip(values),
    }


def main() -> None:
    sessions = load_data()
    rows: list[dict] = []
    session_inventory = []

    for session_index, session in enumerate(sessions):
        response = np.asarray(session["response"])
        choice_valid = np.isin(response, [-1, 1])
        choice = (response[choice_valid] == 1).astype(int)
        stimulus_all = np.sign(
            np.asarray(session["contrast_right"]) - np.asarray(session["contrast_left"])
        ).astype(int)
        stimulus = stimulus_all[choice_valid]
        correct = np.asarray(session["feedback_type"])[choice_valid] == 1
        error = np.asarray(session["feedback_type"])[choice_valid] == -1
        difficulty = np.abs(
            np.asarray(session["contrast_right"])[choice_valid]
            - np.asarray(session["contrast_left"])[choice_valid]
        )
        reaction_ms = np.asarray(session["reaction_time"])[choice_valid, 0].astype(float)

        inventory = {
            "session": session_index,
            "mouse": session["mouse_name"],
            "date": session["date_exp"],
            "choice_trials": int(choice_valid.sum()),
            "correct_unequal": int(np.sum(correct & (stimulus != 0))),
            "error_unequal": int(np.sum(error & (stimulus != 0))),
            "equal_contrast": int(np.sum(stimulus == 0)),
        }
        session_inventory.append(inventory)

        masks_choice = {
            "all": np.ones(len(choice), dtype=bool),
            "correct": correct,
            "error": error,
            "hard": np.isclose(difficulty, 0.25),
            "medium": np.isclose(difficulty, 0.50),
            "easy": difficulty >= 0.75,
        }

        for window_name, window in WINDOWS.items():
            built = feature_matrices(session, window)
            if built is None:
                continue
            local_all, inter_all, subsets = built
            local = local_all[choice_valid]
            inter = inter_all[choice_valid]

            # Choice decoding, including correctness/difficulty and pre-movement controls.
            window_masks = dict(masks_choice)
            if window_name == "post_early":
                window_masks["pre_movement"] = reaction_ms >= 250
            elif window_name == "post_late":
                window_masks["pre_movement"] = reaction_ms >= 500
            local_scores = repeated_scores(local, choice, window_masks)
            inter_scores = repeated_scores(inter, choice, window_masks, subsets)
            for stratum in window_masks:
                for feature, values in (("local", local_scores), ("inter", inter_scores)):
                    rows.append(
                        {
                            "session": session_index,
                            "mouse": session["mouse_name"],
                            "window": window_name,
                            "outcome": f"choice_{window_name}_{stratum}",
                            "feature": feature,
                            "value": values[stratum],
                        }
                    )

            # Choice decoding when the two sides had equal contrast.
            equal = stimulus == 0
            for feature, X, sels in (("local", local, None), ("inter", inter, subsets)):
                if np.sum(equal) >= MIN_TRIALS and len(np.unique(choice[equal])) == 2:
                    score = repeated_scores(
                        X[equal], choice[equal], {"equal": np.ones(np.sum(equal), bool)}, sels
                    )["equal"]
                else:
                    score = float("nan")
                rows.append(
                    {
                        "session": session_index,
                        "mouse": session["mouse_name"],
                        "window": window_name,
                        "outcome": f"choice_{window_name}_equal_contrast",
                        "feature": feature,
                        "value": score,
                    }
                )

            # Direct stimulus-side decoding on unequal-contrast trials.
            unequal = stimulus != 0
            stim_y = (stimulus[unequal] > 0).astype(int)
            for feature, X, sels in (("local", local, None), ("inter", inter, subsets)):
                score = repeated_scores(
                    X[unequal], stim_y, {"all": np.ones(np.sum(unequal), bool)}, sels
                )["all"]
                rows.append(
                    {
                        "session": session_index,
                        "mouse": session["mouse_name"],
                        "window": window_name,
                        "outcome": f"stimulus_{window_name}_unequal",
                        "feature": feature,
                        "value": score,
                    }
                )

            # Correct-to-error stimulus generalization disambiguates stimulus from choice.
            for feature, X, sels in (("local", local, None), ("inter", inter, subsets)):
                score = train_correct_test_error(X, stimulus, correct, error, sels)
                rows.append(
                    {
                        "session": session_index,
                        "mouse": session["mouse_name"],
                        "window": window_name,
                        "outcome": f"stimulus_generalization_{window_name}_correct_to_error",
                        "feature": feature,
                        "value": score,
                    }
                )

        print(f"processed session {session_index + 1}/{len(sessions)}", flush=True)

    primary_outcomes = [f"choice_{w}_all" for w in WINDOWS]
    stimulus_outcomes = [f"stimulus_{w}_unequal" for w in WINDOWS]
    equal_outcomes = [f"choice_{w}_equal_contrast" for w in WINDOWS]
    generalization_outcomes = [
        f"stimulus_generalization_{w}_correct_to_error" for w in WINDOWS
    ]
    difficulty_outcomes = [
        f"choice_{w}_{level}"
        for w in ("post_early", "post_late")
        for level in ("hard", "medium", "easy")
    ]
    correctness_outcomes = [
        f"choice_{w}_{status}"
        for w in ("post_early", "post_late")
        for status in ("correct", "error")
    ]
    movement_outcomes = [
        "choice_post_early_pre_movement",
        "choice_post_late_pre_movement",
    ]

    temporal_contrasts = add_holm_to_contrasts([
        paired_contrast(rows, "choice_post_early_all", "choice_pre_late_all", "local"),
        paired_contrast(rows, "choice_post_late_all", "choice_pre_late_all", "local"),
        paired_contrast(rows, "choice_post_early_all", "choice_pre_late_all", "inter"),
        paired_contrast(rows, "choice_post_late_all", "choice_pre_late_all", "inter"),
        interaction_contrast(rows, "post_early"),
        interaction_contrast(rows, "post_late"),
    ])
    review_control_contrasts = add_holm_to_contrasts([
        paired_contrast(rows, "choice_post_early_correct", "choice_post_early_error", "local"),
        paired_contrast(rows, "choice_post_late_correct", "choice_post_late_error", "local"),
        paired_contrast(rows, "choice_post_early_easy", "choice_post_early_hard", "local"),
        paired_contrast(rows, "choice_post_late_easy", "choice_post_late_hard", "local"),
    ])

    results = {
        "analysis_parameters": {
            "sessions": len(sessions),
            "mice": len({s["mouse_name"] for s in sessions}),
            "minimum_neurons_per_region": MIN_NEURONS,
            "pair_subsets": N_PAIR_SUBSETS,
            "score": "balanced_accuracy",
            "inference_unit": "mouse",
        },
        "inventory": session_inventory,
        "primary": summarize(rows, primary_outcomes),
        "stimulus": summarize(rows, stimulus_outcomes),
        "equal_contrast": summarize(rows, equal_outcomes),
        "correct_to_error": summarize(rows, generalization_outcomes),
        "difficulty": summarize(rows, difficulty_outcomes),
        "correctness": summarize(rows, correctness_outcomes),
        "pre_movement": summarize(rows, movement_outcomes),
        "temporal_contrasts": temporal_contrasts,
        "review_control_contrasts": review_control_contrasts,
        "session_rows": rows,
    }

    (OUT_DIR / "reanalysis_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # Compact figure used in the revised manuscript.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"local": "#d95f02", "inter": "#1b9e77"}
    labels = list(WINDOWS)

    for feature in ("inter", "local"):
        means = []
        lows = []
        highs = []
        for outcome in primary_outcomes:
            values = np.array(list(aggregate_by_mouse(rows, outcome, feature).values()))
            means.append(values.mean())
            lo, hi = bootstrap_ci(values)
            lows.append(values.mean() - lo)
            highs.append(hi - values.mean())
        axes[0, 0].errorbar(
            range(4), means, yerr=[lows, highs], marker="o", capsize=4,
            label=feature.replace("inter", "inter-area").title(), color=colors[feature]
        )
    axes[0, 0].axhline(0.5, color="gray", ls="--", lw=1)
    axes[0, 0].set_xticks(range(4), ["Pre early", "Pre late", "Post early", "Post late"])
    axes[0, 0].set_ylabel("Balanced accuracy")
    axes[0, 0].set_title("A. Choice decoding across time")
    axes[0, 0].legend(frameon=False)

    for ax, family, title in (
        (axes[0, 1], stimulus_outcomes, "B. Stimulus-side decoding"),
        (axes[1, 0], equal_outcomes, "C. Choice decoding, equal contrasts"),
        (axes[1, 1], generalization_outcomes, "D. Correct-to-error stimulus generalization"),
    ):
        x = np.arange(4)
        width = 0.34
        for offset, feature in ((-width / 2, "inter"), (width / 2, "local")):
            vals = []
            err_lo = []
            err_hi = []
            for outcome in family:
                v = np.array(list(aggregate_by_mouse(rows, outcome, feature).values()))
                vals.append(v.mean())
                lo, hi = bootstrap_ci(v)
                err_lo.append(v.mean() - lo)
                err_hi.append(hi - v.mean())
            ax.bar(x + offset, vals, width, color=colors[feature], alpha=0.88,
                   label=feature.replace("inter", "inter-area").title())
            ax.errorbar(x + offset, vals, yerr=[err_lo, err_hi], fmt="none",
                        ecolor="black", capsize=3, lw=1)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_xticks(x, ["Pre early", "Pre late", "Post early", "Post late"], rotation=15)
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "reanalysis_controls.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure focused on the review-requested difficulty and correct/error controls.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panel_specs = [
        (
            axes[0],
            difficulty_outcomes,
            ["Early hard", "Early medium", "Early easy", "Late hard", "Late medium", "Late easy"],
            "A. Choice decoding by stimulus difficulty",
        ),
        (
            axes[1],
            correctness_outcomes,
            ["Early correct", "Early error", "Late correct", "Late error"],
            "B. Choice decoding by behavioral outcome",
        ),
    ]
    for ax, family, tick_labels, title in panel_specs:
        x = np.arange(len(family))
        width = 0.36
        for offset, feature in ((-width / 2, "inter"), (width / 2, "local")):
            values, lo_err, hi_err = [], [], []
            for outcome in family:
                v = np.array(list(aggregate_by_mouse(rows, outcome, feature).values()))
                mean = v.mean()
                lo, hi = bootstrap_ci(v)
                values.append(mean)
                lo_err.append(mean - lo)
                hi_err.append(hi - mean)
            ax.bar(x + offset, values, width, color=colors[feature], alpha=0.88,
                   label=feature.replace("inter", "inter-area").title())
            ax.errorbar(x + offset, values, yerr=[lo_err, hi_err], fmt="none",
                        ecolor="black", capsize=3, lw=1)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_xticks(x, tick_labels, rotation=20, ha="right")
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reanalysis_difficulty_errors.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({k: v for k, v in results.items() if k not in {"inventory", "session_rows"}}, indent=2))


if __name__ == "__main__":
    main()

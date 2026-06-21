"""Statistical analysis for the combined LLM eval (Part A + Part B).

Produces the six deliverables into results/summary_llm/:
  1. llm_main_results_long.csv   - Part A, one row per evaluated cell
  2. llm_main_significance.csv   - Part A paired/unpaired tests (Holm-adjusted)
  3. sysid_results_long.csv      - Part B, one row per cell (incl. inferred consts)
  4. sysid_significance.csv      - Part B matched hidden-disclosed paired tests
  5. sysid_inference.csv         - per model x constant inference MAE/correlation
                                   + accuracy-vs-inference link
  6. llm_summary.md              - per-claim verdict + CI + Holm p + total cost

Scoring: per-cell error = mean over links of |wrap(theta_pred - theta_true)|,
already computed in the checkpoints. Reliability-adjusted: a failed/unparseable
cell is scored at pi/2 (random-guess baseline), not dropped; raw success
reported separately. Holm correction within each test family.

Cost is a best-effort estimate from real token counts x public per-Mtoken prices.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

PI_2 = math.pi / 2.0
N_BOOT = 10000
RNG = np.random.default_rng(770021)

PRICING = {  # USD per 1M tokens (estimate only; Azure billing may differ)
    "kimi-k2.6":               {"input": 0.60, "output": 2.50},
    "deepseek-v4-pro":         {"input": 0.27, "output": 1.10},
    "grok-4-1-fast-reasoning": {"input": 0.20, "output": 0.50},
}


# ----------------------------------------------------------------- loaders ---
def _adj_err(rec):
    ok = bool(rec.get("success"))
    ae = rec.get("angle_error_mean")
    if ok and ae is not None and np.isfinite(ae):
        return float(ae), 1
    return PI_2, 0


def load_part_a(ckpt_dir):
    rows = []
    for p in glob.glob(os.path.join(ckpt_dir, "*.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        err, ok = _adj_err(r)
        rows.append({
            "model": r.get("model"), "k": r.get("k"), "regime": r.get("regime"),
            "horizon": r.get("horizon"), "prompting": r.get("prompting"),
            "movement_id": r.get("movement_id"), "success": ok,
            "angle_error_mean": err,
            "prompt_tokens": r.get("prompt_tokens") or 0,
            "completion_tokens": r.get("completion_tokens") or 0,
            "latency_s": r.get("latency_s") or 0.0,
        })
    return pd.DataFrame(rows)


def load_part_b(ckpt_dir):
    rows = []
    for p in glob.glob(os.path.join(ckpt_dir, "*.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        err, ok = _adj_err(r)
        inf = r.get("inferred_constants") or {}
        tc = r.get("true_constants") or {}
        rows.append({
            "model": r.get("model"), "k": r.get("k"),
            "constant_set": r.get("set_id"), "horizon": r.get("horizon"),
            "disclosure": r.get("disclosure"), "trajectory_id": r.get("movement_id"),
            "success": ok, "angle_error_mean": err,
            "difficulty": r.get("difficulty"),
            "true_g": tc.get("g"), "true_L": tc.get("L"), "true_m": tc.get("m"),
            "true_damping": tc.get("damping"),
            "inf_g": inf.get("g"), "inf_L": inf.get("L"), "inf_m": inf.get("m"),
            "inf_damping": inf.get("damping"),
            "prompt_tokens": r.get("prompt_tokens") or 0,
            "completion_tokens": r.get("completion_tokens") or 0,
            "latency_s": r.get("latency_s") or 0.0,
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------- statistics ---
def boot_ci_mean(x, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    bm = x[idx].mean(axis=1)
    return (float(np.percentile(bm, 100 * alpha / 2)),
            float(np.percentile(bm, 100 * (1 - alpha / 2))))


def boot_ci_diff_indep(a, b, alpha=0.05):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    ia = RNG.integers(0, len(a), size=(N_BOOT, len(a)))
    ib = RNG.integers(0, len(b), size=(N_BOOT, len(b)))
    bd = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return (float(np.percentile(bd, 100 * alpha / 2)),
            float(np.percentile(bd, 100 * (1 - alpha / 2))))


def rank_biserial_paired(diffs):
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    return float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())


def paired_test(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    diffs = a - b
    n = len(diffs)
    if n < 5:
        return None
    lo, hi = boot_ci_mean(diffs)
    if np.all(diffs == 0):
        p = 1.0
    else:
        try:
            _, p = stats.wilcoxon(a, b, zero_method="wilcox",
                                  alternative="two-sided")
        except Exception:
            p = float("nan")
    return {"mean_diff": float(diffs.mean()), "ci_lo": lo, "ci_hi": hi,
            "p_raw": float(p), "n": n, "effect_size": rank_biserial_paired(diffs),
            "paired": 1}


def unpaired_test(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 5 or len(b) < 5:
        return None
    lo, hi = boot_ci_diff_indep(a, b)
    try:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        eff = 2.0 * u / (len(a) * len(b)) - 1.0  # rank-biserial for MWU
    except Exception:
        p, eff = float("nan"), float("nan")
    return {"mean_diff": float(a.mean() - b.mean()), "ci_lo": lo, "ci_hi": hi,
            "p_raw": float(p), "n": len(a) + len(b), "effect_size": float(eff),
            "paired": 0}


def holm(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (n - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def _pivot_paired(df, keys, split_col, left, right):
    sub = df[df[split_col].isin([left, right])]
    g = sub.groupby(keys + [split_col])["angle_error_mean"].mean().reset_index()
    p = g.pivot_table(index=keys, columns=split_col, values="angle_error_mean")
    if left not in p.columns or right not in p.columns:
        return np.array([]), np.array([])
    p = p.dropna(subset=[left, right])
    return p[left].to_numpy(), p[right].to_numpy()


# ----------------------------------------------------------- Part A tests ----
def part_a_significance(df):
    tests = []
    models = sorted(df["model"].dropna().unique())
    ks = sorted(df["k"].dropna().unique())

    # regime contrasts: UNPAIRED (ICs differ across regimes -> confounded;
    # the clean matched test lives in Part B). Per k and overall.
    for comp, (left, right) in [
        ("hidden_minus_disclosed", ("changed_hidden", "changed_disclosed")),
        ("hidden_minus_normal", ("changed_hidden", "normal")),
    ]:
        for k in ks:
            sub = df[df["k"] == k]
            a = sub[sub["regime"] == left]["angle_error_mean"].to_numpy()
            b = sub[sub["regime"] == right]["angle_error_mean"].to_numpy()
            r = unpaired_test(a, b)
            if r:
                tests.append({"comparison": comp, "stratum": f"k{k}", **r})
        a = df[df["regime"] == left]["angle_error_mean"].to_numpy()
        b = df[df["regime"] == right]["angle_error_mean"].to_numpy()
        r = unpaired_test(a, b)
        if r:
            tests.append({"comparison": comp, "stratum": "overall", **r})

    # CoT vs no_cot: PAIRED on (movement_id, horizon) within model + overall.
    for m in models:
        sub = df[df["model"] == m]
        a, b = _pivot_paired(sub, ["movement_id", "horizon"],
                             "prompting", "cot", "no_cot")
        r = paired_test(a, b)
        if r:
            tests.append({"comparison": "cot_minus_nocot", "stratum": m, **r})
    a, b = _pivot_paired(df, ["model", "movement_id", "horizon"],
                         "prompting", "cot", "no_cot")
    r = paired_test(a, b)
    if r:
        tests.append({"comparison": "cot_minus_nocot", "stratum": "overall", **r})

    # model vs model: PAIRED on (movement_id, horizon, prompting).
    for m1, m2 in combinations(models, 2):
        a, b = _pivot_paired(df, ["movement_id", "horizon", "prompting"],
                             "model", m1, m2)
        r = paired_test(a, b)
        if r:
            tests.append({"comparison": f"{m1}_minus_{m2}",
                          "stratum": "overall", **r})

    out = pd.DataFrame(tests)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_raw"].fillna(1.0).to_numpy())
    return out[["comparison", "stratum", "mean_diff", "ci_lo", "ci_hi",
                "p_raw", "p_holm", "n", "effect_size", "paired"]]


# ----------------------------------------------------------- Part B tests ----
def part_b_significance(df):
    tests = []
    ks = sorted(df["k"].dropna().unique())
    horizons = sorted(df["horizon"].dropna().unique())
    keys = ["model", "trajectory_id", "horizon"]  # same traj, disclosure toggled
    # overall
    a, b = _pivot_paired(df, keys, "disclosure", "hidden", "disclosed")
    r = paired_test(a, b)
    if r:
        tests.append({"comparison": "hidden_minus_disclosed",
                      "stratum": "overall", **r})
    # per k
    for k in ks:
        a, b = _pivot_paired(df[df["k"] == k], keys, "disclosure",
                             "hidden", "disclosed")
        r = paired_test(a, b)
        if r:
            tests.append({"comparison": "hidden_minus_disclosed",
                          "stratum": f"k{k}", **r})
    # per horizon
    for h in horizons:
        a, b = _pivot_paired(df[df["horizon"] == h], keys, "disclosure",
                             "hidden", "disclosed")
        r = paired_test(a, b)
        if r:
            tests.append({"comparison": "hidden_minus_disclosed",
                          "stratum": f"h{h:g}", **r})
    out = pd.DataFrame(tests)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_raw"].fillna(1.0).to_numpy())
    return out[["comparison", "stratum", "mean_diff", "ci_lo", "ci_hi",
                "p_raw", "p_holm", "n", "effect_size", "paired"]]


# ----------------------------------------------------- Part B inference probe -
def _spearman(x, y):
    if len(x) < 5:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def _pearson(x, y):
    if len(x) < 5:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _boot_corr_ci(x, y, kind="spearman"):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 5:
        return float("nan"), float("nan")
    out = []
    for _ in range(2000):
        idx = RNG.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        if kind == "spearman":
            out.append(stats.spearmanr(xb, yb)[0])
        else:
            out.append(stats.pearsonr(xb, yb)[0])
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _is_num(x):
    # pandas turns None into np.nan in numeric columns, so a plain
    # "is not None" check lets NaN through; require an actual finite number.
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def part_b_inference(df):
    """Per model x constant: MAE + Spearman/Pearson(inferred,true) with CIs.
    Plus per-model accuracy-vs-inference correlation across trajectories."""
    hid = df[df["disclosure"] == "hidden"].copy()
    rows = []
    models = sorted(hid["model"].dropna().unique())
    for m in models:
        sub = hid[hid["model"] == m]
        # scalar constants
        for const in ("g", "damping"):
            pairs = [(float(r[f"inf_{const}"]), float(r[f"true_{const}"]))
                     for _, r in sub.iterrows()
                     if _is_num(r[f"inf_{const}"]) and _is_num(r[f"true_{const}"])]
            rows.append(_inf_row(m, const, pairs))
        # vector constants -> pool across link positions
        for const in ("L", "m"):
            pairs = []
            for _, r in sub.iterrows():
                iv, tv = r[f"inf_{const}"], r[f"true_{const}"]
                if isinstance(iv, list) and isinstance(tv, list):
                    for a, b in zip(iv, tv):
                        if _is_num(a) and _is_num(b):
                            pairs.append((float(a), float(b)))
            rows.append(_inf_row(m, const, pairs))
    inf_df = pd.DataFrame(rows)

    # accuracy-vs-inference: per cell, normalized constant-inference error vs
    # forecast error; Spearman across all hidden cells per model.
    link_rows = []
    for m in models:
        sub = hid[hid["model"] == m]
        cerr, ferr = [], []
        for _, r in sub.iterrows():
            e = _const_infer_error(r)
            if e is not None and np.isfinite(r["angle_error_mean"]):
                cerr.append(e); ferr.append(r["angle_error_mean"])
        if len(cerr) >= 5:
            rho, p = _spearman(cerr, ferr)
            lo, hi = _boot_corr_ci(cerr, ferr, "spearman")
            link_rows.append({"model": m, "constant": "ACCURACY_VS_INFERENCE",
                              "n": len(cerr), "mae": float("nan"),
                              "spearman": rho, "spearman_p": p,
                              "spearman_ci_lo": lo, "spearman_ci_hi": hi,
                              "pearson": float("nan"), "pearson_p": float("nan")})
    if link_rows:
        inf_df = pd.concat([inf_df, pd.DataFrame(link_rows)], ignore_index=True)
    return inf_df


def _const_infer_error(r):
    """Mean relative |inferred-true| over all available constants for one cell."""
    errs = []
    for const in ("g", "damping"):
        iv, tv = r[f"inf_{const}"], r[f"true_{const}"]
        if _is_num(iv) and _is_num(tv):
            iv, tv = float(iv), float(tv)
            denom = abs(tv) if abs(tv) > 1e-6 else 1.0
            errs.append(abs(iv - tv) / denom)
    for const in ("L", "m"):
        iv, tv = r[f"inf_{const}"], r[f"true_{const}"]
        if isinstance(iv, list) and isinstance(tv, list):
            for a, b in zip(iv, tv):
                if _is_num(a) and _is_num(b):
                    a, b = float(a), float(b)
                    denom = abs(b) if abs(b) > 1e-6 else 1.0
                    errs.append(abs(a - b) / denom)
    return float(np.mean(errs)) if errs else None


def _inf_row(model, const, pairs):
    if not pairs:
        return {"model": model, "constant": const, "n": 0, "mae": float("nan"),
                "spearman": float("nan"), "spearman_p": float("nan"),
                "spearman_ci_lo": float("nan"), "spearman_ci_hi": float("nan"),
                "pearson": float("nan"), "pearson_p": float("nan")}
    inf = np.array([p[0] for p in pairs], float)
    tru = np.array([p[1] for p in pairs], float)
    mae = float(np.abs(inf - tru).mean())
    rho, sp = _spearman(inf, tru)
    pr, pp = _pearson(inf, tru)
    lo, hi = _boot_corr_ci(inf, tru, "spearman")
    return {"model": model, "constant": const, "n": len(pairs), "mae": mae,
            "spearman": rho, "spearman_p": sp,
            "spearman_ci_lo": lo, "spearman_ci_hi": hi,
            "pearson": pr, "pearson_p": pp}


# ----------------------------------------------------------------- cost -------
def total_cost(*dfs):
    total = 0.0
    for df in dfs:
        if df is None or df.empty:
            continue
        for m, sub in df.groupby("model"):
            pr = PRICING.get(m, {"input": 0.0, "output": 0.0})
            total += (sub["prompt_tokens"].sum() * pr["input"]
                      + sub["completion_tokens"].sum() * pr["output"]) / 1e6
    return total


# --------------------------------------------------------------- markdown -----
def _df_to_md(df):
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def _verdict(row):
    sig = "**significant**" if row["p_holm"] < 0.05 else "n.s."
    direction = ""
    if row["p_holm"] < 0.05:
        direction = (" (lower error for "
                     + ("first" if row["mean_diff"] < 0 else "second") + " term)")
    kind = "paired" if row.get("paired", 1) else "unpaired"
    return (f"- `{row['comparison']}` [{row['stratum']}, {kind}]: "
            f"mean diff {row['mean_diff']:+.4f} rad "
            f"(95% CI [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]), "
            f"r={row['effect_size']:+.3f}, n={row['n']}, "
            f"p_holm={row['p_holm']:.4f} -> {sig}{direction}")


def write_summary(path, dfa, sa, dfb, sb, inf, cost):
    L = ["# Combined LLM physics-forecasting: statistical results\n"]
    L.append(f"- Part A cells: **{len(dfa)}** (success {dfa['success'].mean()*100:.1f}%)"
             f"  |  Part B cells: **{len(dfb)}** (success {dfb['success'].mean()*100:.1f}%)")
    L.append("- Reliability-adjusted error (failed cell = pi/2 rad). "
             "Paired Wilcoxon / unpaired Mann-Whitney; 95% CIs by bootstrap "
             f"(B={N_BOOT}); p_holm = Holm within each family.")
    L.append(f"- **Total estimated API cost: ${cost:.2f}**\n")

    L.append("## Part A - cross-model ranking (paired)")
    for _, r in sa[sa["comparison"].str.contains("_minus_")
                   & ~sa["comparison"].str.startswith("hidden_minus")
                   & (sa["comparison"] != "cot_minus_nocot")].iterrows():
        L.append(_verdict(r))
    L.append("\n## Part A - chain-of-thought (cot - no_cot, paired)")
    for _, r in sa[sa["comparison"] == "cot_minus_nocot"].iterrows():
        L.append(_verdict(r))
    L.append("\n## Part A - regime contrasts (UNPAIRED; confounded - see Part B)")
    L.append("Negative diff = changed_hidden has LOWER error. ICs differ across "
             "regimes, so disclosure and constants are conflated here.")
    for _, r in sa[sa["comparison"].str.startswith("hidden_minus")].iterrows():
        L.append(_verdict(r))

    L.append("\n## Part B - matched hidden-disclosed (clean paired contrast)")
    L.append("SAME trajectories, only disclosure toggled. Negative diff = "
             "hidden has LOWER error (in-context system-ID benefit).")
    for _, r in sb.iterrows():
        L.append(_verdict(r))

    L.append("\n## Part B - inference probe (inferred vs true constants)")
    L.append("Per model x constant: MAE and Spearman(inferred, true) with 95% CI. "
             "L/m pooled across link positions. A `nan` Spearman means the model "
             "emitted a CONSTANT value for that constant (no variation -> "
             "correlation undefined): i.e. it fell back on an Earth-standard prior "
             "(g~9.81, L=m=1.0) instead of inferring. ACCURACY_VS_INFERENCE = "
             "Spearman(constant-inference error, forecast error): positive => "
             "better system-ID tracks lower forecast error.")
    show = inf[["model", "constant", "n", "mae", "spearman",
                "spearman_ci_lo", "spearman_ci_hi", "spearman_p", "pearson"]]
    L.append(_df_to_md(show))

    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part-a-dir", default="results/llm_main")
    ap.add_argument("--part-b-dir", default="results/llm_sysid")
    ap.add_argument("--out-dir", default="results/summary_llm")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dfa = load_part_a(args.part_a_dir)
    dfb = load_part_b(args.part_b_dir)
    if dfa.empty and dfb.empty:
        print("No checkpoints found in either directory.")
        return

    a_cols = ["model", "k", "regime", "horizon", "prompting", "movement_id",
              "success", "angle_error_mean", "prompt_tokens",
              "completion_tokens", "latency_s"]
    dfa[a_cols].to_csv(os.path.join(args.out_dir, "llm_main_results_long.csv"),
                       index=False)
    sa = part_a_significance(dfa)
    sa.to_csv(os.path.join(args.out_dir, "llm_main_significance.csv"), index=False)

    b_cols = ["model", "constant_set", "k", "horizon", "disclosure",
              "trajectory_id", "success", "angle_error_mean", "difficulty",
              "inf_g", "inf_L", "inf_m", "inf_damping",
              "true_g", "true_L", "true_m", "true_damping",
              "prompt_tokens", "completion_tokens", "latency_s"]
    dfb[b_cols].to_csv(os.path.join(args.out_dir, "sysid_results_long.csv"),
                       index=False)
    sb = part_b_significance(dfb)
    sb.to_csv(os.path.join(args.out_dir, "sysid_significance.csv"), index=False)
    inf = part_b_inference(dfb)
    inf.to_csv(os.path.join(args.out_dir, "sysid_inference.csv"), index=False)

    cost = total_cost(dfa, dfb)
    write_summary(os.path.join(args.out_dir, "llm_summary.md"),
                  dfa, sa, dfb, sb, inf, cost)

    print(f"Wrote 6 files to {args.out_dir}/")
    print(f"  Part A: {len(dfa)} cells, {dfa['success'].mean()*100:.1f}% success, "
          f"{len(sa)} tests")
    print(f"  Part B: {len(dfb)} cells, {dfb['success'].mean()*100:.1f}% success, "
          f"{len(sb)} tests")
    print(f"  Total estimated cost: ${cost:.2f}")


if __name__ == "__main__":
    main()

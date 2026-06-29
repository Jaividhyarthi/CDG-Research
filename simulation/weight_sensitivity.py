"""
CDG Weight Sensitivity Analysis — simulation/weight_sensitivity.py
Tests 200+ weight combinations against simulation data.
Proves proposed weights are at or near the empirical optimum.
This directly answers the reviewer challenge: "weights appear arbitrary."
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from itertools import product
from tqdm import tqdm
import os
import glob

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'outputs')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')

# ── Proposed weights (from CDG foundation document) ───────────────────────────
PROPOSED = {
    'PQAR': 0.20,
    'QCS':  0.15,
    'TTQ':  0.10,
    'ARWM': 0.25,
    'RET':  0.15,
    'OCR':  0.15,
}

SIGNAL_COLS = ['PQAR_n', 'QCS_n', 'TTQ_n', 'ARWM_n', 'RET_n', 'OCR_n']
SIGNAL_KEYS = ['PQAR', 'QCS', 'TTQ', 'ARWM', 'RET', 'OCR']

plt.rcParams.update({
    'figure.facecolor': '#0b0f1a',
    'axes.facecolor':   '#111827',
    'axes.edgecolor':   '#1e2d45',
    'axes.labelcolor':  '#e2e8f0',
    'xtick.color':      '#94a3b8',
    'ytick.color':      '#94a3b8',
    'text.color':       '#e2e8f0',
    'grid.color':       '#1e2d45',
    'grid.linewidth':   0.5,
    'font.size':        10,
})


# ── Load session data ─────────────────────────────────────────────────────────
def load_sessions():
    files = glob.glob(os.path.join(DATA_DIR, 'SYN_*.csv'))
    all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Get final tick per session
    final = all_df.groupby('session_id').last().reset_index()
    final['gt_binary'] = (final['profile'] != 'autonomous').astype(int)
    return all_df, final


# ── Compute CDG with arbitrary weights ────────────────────────────────────────
def cdg_with_weights(row_signals, weights, sfi):
    """Compute CDG inner sum with given weights, apply SFI."""
    inner = sum(weights[k] * row_signals[i] for i, k in enumerate(SIGNAL_KEYS))
    return min(1.0, max(0.0, sfi * inner))


def evaluate_weights(weights, final_df):
    """Compute AUC for a given weight dict against session final states."""
    scores = []
    for _, row in final_df.iterrows():
        sigs = [row[c] for c in SIGNAL_COLS]
        sfi  = row['SFI']
        cdg  = cdg_with_weights(sigs, weights, sfi)
        scores.append(cdg)
    try:
        return roc_auc_score(final_df['gt_binary'], scores)
    except Exception:
        return 0.0


# ── Generate weight combinations ──────────────────────────────────────────────
def generate_weight_combinations(n_samples=300):
    """
    Sample random weight combinations that sum to 1.0.
    Uses Dirichlet distribution for uniform coverage of the simplex.
    Also includes the proposed weights as a fixed entry.
    """
    rng = np.random.default_rng(42)
    combos = []

    # Always include proposed weights first
    combos.append(list(PROPOSED.values()))

    # Sample random combinations via Dirichlet
    samples = rng.dirichlet(np.ones(6), size=n_samples - 1)
    combos.extend(samples.tolist())

    return combos


# ── Run sensitivity analysis ──────────────────────────────────────────────────
def run_sensitivity(final_df, n_samples=300):
    combos = generate_weight_combinations(n_samples)
    results = []

    print(f"Testing {len(combos)} weight combinations...")

    for i, combo in enumerate(tqdm(combos, desc="Weight combos")):
        w = dict(zip(SIGNAL_KEYS, combo))
        auc = evaluate_weights(w, final_df)
        results.append({
            'combo_id':   i,
            'is_proposed': i == 0,
            'PQAR': combo[0], 'QCS': combo[1], 'TTQ': combo[2],
            'ARWM': combo[3], 'RET': combo[4], 'OCR': combo[5],
            'AUC':  auc,
        })

    return pd.DataFrame(results)


# ── Figure 6: Weight sensitivity scatter ─────────────────────────────────────
def plot_sensitivity(results_df):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor('#0b0f1a')
    fig.suptitle('Figure 6 — Weight Sensitivity Analysis: AUC vs Individual Weights',
                 fontsize=12, y=1.01)

    axes = axes.flatten()
    proposed_auc = results_df[results_df['is_proposed']]['AUC'].values[0]

    for i, key in enumerate(SIGNAL_KEYS):
        ax = axes[i]
        # All combinations
        ax.scatter(results_df[key], results_df['AUC'],
                   color='#475569', alpha=0.35, s=18, label='Random combos')
        # Proposed weight point
        proposed_val = PROPOSED[key]
        ax.scatter([proposed_val], [proposed_auc],
                   color='#8b5cf6', s=120, zorder=5,
                   marker='*', label=f'Proposed (w={proposed_val})')
        ax.axhline(proposed_auc, color='#8b5cf6', linestyle='--',
                   linewidth=0.8, alpha=0.5)
        ax.set_xlabel(f'Weight — {key}')
        ax.set_ylabel('AUC')
        ax.set_title(f'{key} (proposed={proposed_val})', fontsize=10)
        ax.legend(fontsize=7, framealpha=0.2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.05)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig6_weight_sensitivity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 7: AUC distribution ───────────────────────────────────────────────
def plot_auc_distribution(results_df):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0b0f1a')

    proposed_auc = results_df[results_df['is_proposed']]['AUC'].values[0]
    random_aucs  = results_df[~results_df['is_proposed']]['AUC']

    ax.hist(random_aucs, bins=40, color='#475569', alpha=0.7,
            edgecolor='none', label='Random weight combinations')
    ax.axvline(proposed_auc, color='#8b5cf6', linewidth=2.5,
               label=f'Proposed weights (AUC={proposed_auc:.4f})')
    ax.axvline(random_aucs.quantile(0.95), color='#f59e0b',
               linewidth=1.5, linestyle='--',
               label=f'95th percentile ({random_aucs.quantile(0.95):.4f})')

    ax.set_xlabel('AUC Score')
    ax.set_ylabel('Number of weight combinations')
    ax.set_title('Figure 7 — AUC Distribution Across 300 Weight Combinations', pad=12)
    ax.legend(framealpha=0.2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig7_auc_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading session data...")
    all_df, final_df = load_sessions()
    print(f"Loaded {len(final_df)} sessions\n")

    results_df = run_sensitivity(final_df, n_samples=300)

    proposed_auc = results_df[results_df['is_proposed']]['AUC'].values[0]
    random_aucs  = results_df[~results_df['is_proposed']]['AUC']

    print(f"\n{'='*55}")
    print(f"  WEIGHT SENSITIVITY ANALYSIS RESULTS")
    print(f"{'='*55}")
    print(f"\n  Proposed weights AUC  : {proposed_auc:.4f}")
    print(f"  Random combos mean AUC: {random_aucs.mean():.4f}")
    print(f"  Random combos max AUC : {random_aucs.max():.4f}")
    print(f"  Random combos min AUC : {random_aucs.min():.4f}")
    print(f"  95th percentile AUC   : {random_aucs.quantile(0.95):.4f}")
    pct_better = (random_aucs > proposed_auc).mean() * 100
    print(f"\n  % of random combos beating proposed: {pct_better:.1f}%")

    # Percentile rank of proposed weights
    pct_rank = (random_aucs <= proposed_auc).mean() * 100
    print(f"  Proposed weights percentile rank   : {pct_rank:.1f}th")

    print(f"\n  Top 5 weight combinations by AUC:")
    top5 = results_df.nlargest(5, 'AUC')[
        ['combo_id', 'is_proposed', 'PQAR', 'QCS', 'TTQ', 'ARWM', 'RET', 'OCR', 'AUC']
    ]
    print(top5.to_string(index=False))

    print(f"\n── Verdict ──")
    if pct_rank >= 80:
        print(f"  ✓ Proposed weights rank at {pct_rank:.1f}th percentile.")
        print(f"    Weights are near-optimal — reviewer challenge answered.")
    else:
        print(f"  ⚠ Proposed weights rank at {pct_rank:.1f}th percentile.")
        print(f"    Consider updating weights to top-performing combination.")
        best = results_df.nlargest(1, 'AUC').iloc[0]
        print(f"    Best combo: PQAR={best.PQAR:.2f} QCS={best.QCS:.2f} "
              f"TTQ={best.TTQ:.2f} ARWM={best.ARWM:.2f} "
              f"RET={best.RET:.2f} OCR={best.OCR:.2f}")

    print(f"\n── Generating figures ──")
    plot_sensitivity(results_df)
    plot_auc_distribution(results_df)

    # Save results
    out_path = os.path.join(OUTPUT_DIR, 'weight_sensitivity_results.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved: {out_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
"""
CDG Statistical Validation — simulation/validate.py
Computes AUC, RMSE, precision/recall, IP precision.
These are the paper's core validation numbers.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error
)
from sklearn.preprocessing import label_binarize
import os
import glob

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'outputs')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0b0f1a',
    'axes.facecolor':   '#111827',
    'axes.edgecolor':   '#1e2d45',
    'axes.labelcolor':  '#e2e8f0',
    'xtick.color':      '#94a3b8',
    'ytick.color':      '#94a3b8',
    'text.color':       '#e2e8f0',
    'grid.color':       '#1e2d45',
    'grid.linewidth':   0.6,
    'font.family':      'DejaVu Sans',
    'font.size':        10,
})

COLORS = {
    'autonomous':        '#10b981',
    'gradual_drifter':   '#8b5cf6',
    'threshold_drifter': '#f59e0b',
    'alert':             '#ef4444',
    'accent':            '#06b6d4',
}


# ── Load data ─────────────────────────────────────────────────────────────────
def load_summary():
    path = os.path.join(DATA_DIR, 'simulation_summary.csv')
    return pd.read_csv(path)


def load_all_sessions():
    files = glob.glob(os.path.join(DATA_DIR, 'SYN_*.csv'))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# ── Ground truth labeling ─────────────────────────────────────────────────────
def assign_ground_truth(summary):
    """
    Binary: autonomous=0 (non-dependent), drifters=1 (dependent).
    Continuous: autonomous→0.2 expected CDG, gradual→0.7, threshold→0.55.
    """
    summary = summary.copy()
    summary['gt_binary'] = (summary['profile'] != 'autonomous').astype(int)
    gt_continuous = {'autonomous': 0.20, 'gradual_drifter': 0.70, 'threshold_drifter': 0.55}
    summary['gt_continuous'] = summary['profile'].map(gt_continuous)
    return summary


# ── Validation metrics ────────────────────────────────────────────────────────
def compute_auc(summary):
    """AUC-ROC: can CDG final score distinguish dependent from autonomous?"""
    auc = roc_auc_score(summary['gt_binary'], summary['cdg_final'])
    fpr, tpr, thresholds = roc_curve(summary['gt_binary'], summary['cdg_final'])
    return auc, fpr, tpr, thresholds


def compute_rmse(summary):
    """RMSE between CDG mean and expected ground-truth continuous value."""
    rmse = np.sqrt(mean_squared_error(summary['gt_continuous'], summary['cdg_mean']))
    return rmse


def compute_classification_metrics(summary, threshold=0.45):
    """Precision/Recall/F1 at a given CDG threshold for binary classification."""
    pred = (summary['cdg_final'] >= threshold).astype(int)
    p  = precision_score(summary['gt_binary'], pred, zero_division=0)
    r  = recall_score(summary['gt_binary'], pred, zero_division=0)
    f1 = f1_score(summary['gt_binary'], pred, zero_division=0)
    cm = confusion_matrix(summary['gt_binary'], pred)
    return p, r, f1, cm, pred


def compute_ip_precision(summary):
    """
    IP precision: among sessions where IP fired, what fraction were truly dependent?
    IP recall: among truly dependent sessions, what fraction had IP detected?
    """
    ip_fired    = summary['ip_count'] > 0
    truly_dep   = summary['gt_binary'] == 1
    tp = (ip_fired & truly_dep).sum()
    fp = (ip_fired & ~truly_dep).sum()
    fn = (~ip_fired & truly_dep).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall, tp, fp, fn


# ── Figure 1: CDG curves by profile ──────────────────────────────────────────
def plot_cdg_curves(all_sessions):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0b0f1a')

    for profile, color in COLORS.items():
        if profile in ('alert', 'accent'):
            continue
        sessions = all_sessions[all_sessions['profile'] == profile]
        sample_ids = sessions['session_id'].unique()[:8]
        for sid in sample_ids:
            s = sessions[sessions['session_id'] == sid]
            ax.plot(s['t_min'], s['CDG'], color=color, alpha=0.25, linewidth=0.8)
        # Mean curve
        mean_curve = sessions.groupby('t_min')['CDG'].mean()
        ax.plot(mean_curve.index, mean_curve.values,
                color=color, linewidth=2.5, label=profile.replace('_', ' ').title())

    ax.axhline(0.60, color=COLORS['alert'], linestyle='--', linewidth=1.2,
               label='Alert threshold (0.60)')
    ax.fill_between([0, 59], 0.8, 1.0, color=COLORS['alert'], alpha=0.06)
    ax.set_xlabel('Session time (minutes)')
    ax.set_ylabel('CDG Score')
    ax.set_title('Figure 1 — CDG(t) Curves by User Profile', pad=12, fontsize=12)
    ax.legend(loc='upper left', framealpha=0.2)
    ax.set_xlim(0, 59)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig1_cdg_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 2: ROC curve ───────────────────────────────────────────────────────
def plot_roc(fpr, tpr, auc):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0b0f1a')
    ax.plot(fpr, tpr, color='#8b5cf6', linewidth=2.5,
            label=f'CDG ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#475569', linestyle='--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Figure 2 — ROC Curve: CDG Dependency Classification', pad=12)
    ax.legend(loc='lower right', framealpha=0.2)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig2_roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 3: CDG distribution by profile ────────────────────────────────────
def plot_distributions(summary):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    fig.patch.set_facecolor('#0b0f1a')
    metrics = [
        ('cdg_mean',  'Mean CDG'),
        ('cdg_final', 'Final CDG'),
        ('cdg_max',   'Max CDG'),
    ]
    for ax, (col, label) in zip(axes, metrics):
        for profile, color in COLORS.items():
            if profile in ('alert', 'accent'):
                continue
            data = summary[summary['profile'] == profile][col]
            ax.hist(data, bins=25, color=color, alpha=0.65,
                    label=profile.replace('_', ' ').title(), edgecolor='none')
        ax.axvline(0.60, color=COLORS['alert'], linestyle='--',
                   linewidth=1.2, label='Alert (0.60)')
        ax.set_xlabel(label)
        ax.set_ylabel('Sessions')
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, framealpha=0.2)
    fig.suptitle('Figure 3 — CDG Score Distributions by Profile', fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_distributions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 4: Confusion matrix ────────────────────────────────────────────────
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor('#0b0f1a')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Pred: Autonomous', 'Pred: Dependent'],
                yticklabels=['True: Autonomous', 'True: Dependent'],
                ax=ax, linewidths=0.5, linecolor='#1e2d45')
    ax.set_title('Figure 4 — CDG Classification Confusion Matrix', pad=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig4_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 5: SFI amplification effect ───────────────────────────────────────
def plot_sfi_effect(all_sessions):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('#0b0f1a')
    for profile, color in COLORS.items():
        if profile in ('alert', 'accent'):
            continue
        sessions = all_sessions[all_sessions['profile'] == profile]
        mean_sfi = sessions.groupby('t_min')['SFI'].mean()
        ax.plot(mean_sfi.index, mean_sfi.values, color=color,
                linewidth=2.2, label=profile.replace('_', ' ').title())
    ax.axhline(1.0, color='#475569', linestyle=':', linewidth=1)
    ax.axhline(2.0, color=COLORS['alert'], linestyle='--',
               linewidth=1, label='SFI max (2.0)')
    ax.set_xlabel('Session time (minutes)')
    ax.set_ylabel('Session Fatigue Index (SFI)')
    ax.set_title('Figure 5 — SFI Progression Over Session', pad=12)
    ax.legend(framealpha=0.2)
    ax.set_ylim(0.9, 2.1)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig5_sfi_effect.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading simulation data...")
    summary     = load_summary()
    all_sessions = load_all_sessions()
    summary     = assign_ground_truth(summary)

    print(f"Loaded {len(summary)} sessions, {len(all_sessions)} ticks total\n")

    # ── Compute metrics ───────────────────────────────────────────────────────
    auc, fpr, tpr, thresholds = compute_auc(summary)
    rmse                      = compute_rmse(summary)
    precision, recall, f1, cm, pred = compute_classification_metrics(summary)
    ip_prec, ip_rec, tp, fp, fn     = compute_ip_precision(summary)

    # ── Print results ─────────────────────────────────────────────────────────
    print("=" * 55)
    print("  CDG STATISTICAL VALIDATION RESULTS")
    print("=" * 55)

    print(f"\n── Binary Classification (dependent vs autonomous) ──")
    print(f"  AUC-ROC            : {auc:.4f}")
    print(f"  Precision          : {precision:.4f}")
    print(f"  Recall             : {recall:.4f}")
    print(f"  F1 Score           : {f1:.4f}")

    print(f"\n── Continuous Score Accuracy ──")
    print(f"  RMSE (CDG mean vs ground truth) : {rmse:.4f}")

    print(f"\n── Inflection Point Detection ──")
    print(f"  IP Precision       : {ip_prec:.4f}  ({tp} TP, {fp} FP)")
    print(f"  IP Recall          : {ip_rec:.4f}  ({tp} TP, {fn} FN)")

    print(f"\n── Paper threshold assessment ──")
    checks = {
        'AUC ≥ 0.85':           auc >= 0.85,
        'RMSE ≤ 0.15':          rmse <= 0.15,
        'Precision ≥ 0.80':     precision >= 0.80,
        'Recall ≥ 0.80':        recall >= 0.80,
        'IP Precision ≥ 0.75':  ip_prec >= 0.75,
    }
    all_pass = True
    for check, passed in checks.items():
        status = '✓' if passed else '✗'
        print(f"  {status} {check}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  All thresholds met — formula validated for paper submission.")
    else:
        print(f"\n  Some thresholds not met — review formula before submission.")

    print(f"\n── Generating figures ──")
    plot_cdg_curves(all_sessions)
    plot_roc(fpr, tpr, auc)
    plot_distributions(summary)
    plot_confusion_matrix(cm)
    plot_sfi_effect(all_sessions)

    print(f"\n{'='*55}")
    print(f"  Validation complete. Figures saved to paper/figures/")
    print(f"{'='*55}\n")

    # Save metrics to CSV
    metrics = {
        'auc': auc, 'rmse': rmse,
        'precision': precision, 'recall': recall, 'f1': f1,
        'ip_precision': ip_prec, 'ip_recall': ip_rec,
        'n_sessions': len(summary),
    }
    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUTPUT_DIR, 'validation_metrics.csv'), index=False
    )


if __name__ == "__main__":
    main()
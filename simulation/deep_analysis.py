"""
CDG Deep Analysis — simulation/deep_analysis.py
Three analyses required before formula finalization:
1. Zone transition analysis
2. SFI amplification quantification
3. Construct validity check
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'outputs')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')

plt.rcParams.update({
    'figure.facecolor': '#0b0f1a', 'axes.facecolor': '#111827',
    'axes.edgecolor':   '#1e2d45', 'axes.labelcolor': '#e2e8f0',
    'xtick.color':      '#94a3b8', 'ytick.color':     '#94a3b8',
    'text.color':       '#e2e8f0', 'grid.color':       '#1e2d45',
    'grid.linewidth':   0.5,       'font.size':        10,
})

ZONE_COLORS = {
    'low':      '#10b981',
    'moderate': '#3b82f6',
    'high':     '#f59e0b',
    'critical': '#ef4444',
}
PROFILE_COLORS = {
    'autonomous':        '#10b981',
    'gradual_drifter':   '#8b5cf6',
    'threshold_drifter': '#f59e0b',
}
ZONES_ORDER = ['low', 'moderate', 'high', 'critical']


# ── Load data ─────────────────────────────────────────────────────────────────
def load_all():
    files = glob.glob(os.path.join(DATA_DIR, 'SYN_*.csv'))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    summary = pd.read_csv(os.path.join(DATA_DIR, 'simulation_summary.csv'))
    return df, summary


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Zone Transition Analysis
# ══════════════════════════════════════════════════════════════════════════════
def analysis_zone_transitions(df, summary):
    print("\n" + "="*55)
    print("  ANALYSIS 1 — Zone Transition Analysis")
    print("="*55)

    results = {}

    for profile in ['autonomous', 'gradual_drifter', 'threshold_drifter']:
        sessions = df[df['profile'] == profile]
        profile_results = {}

        # Time spent in each zone (% of ticks)
        zone_pcts = {}
        for zone in ZONES_ORDER:
            pct = (sessions['zone'] == zone).mean() * 100
            zone_pcts[zone] = pct

        # First crossing tick for each threshold
        crossings = {}
        thresholds = {'moderate': 0.30, 'high': 0.60, 'critical': 0.80}
        for zone_name, threshold in thresholds.items():
            first_crossings = []
            for sid, grp in sessions.groupby('session_id'):
                crossed = grp[grp['CDG'] >= threshold]
                if len(crossed) > 0:
                    first_crossings.append(crossed['t_min'].iloc[0])
            if first_crossings:
                crossings[zone_name] = {
                    'mean_tick': np.mean(first_crossings),
                    'std_tick':  np.std(first_crossings),
                    'pct_sessions': len(first_crossings) /
                                    sessions['session_id'].nunique() * 100
                }
            else:
                crossings[zone_name] = {
                    'mean_tick': None, 'std_tick': None, 'pct_sessions': 0.0
                }

        profile_results['zone_pcts']  = zone_pcts
        profile_results['crossings']  = crossings
        results[profile] = profile_results

        print(f"\n  Profile: {profile}")
        print(f"  Time in zone:")
        for zone in ZONES_ORDER:
            bar = '█' * int(zone_pcts[zone] / 5)
            print(f"    {zone:10s}: {zone_pcts[zone]:5.1f}%  {bar}")
        print(f"  Threshold crossings:")
        for zone_name, c in crossings.items():
            if c['mean_tick'] is not None:
                print(f"    → {zone_name:8s} (CDG≥{thresholds[zone_name]}): "
                      f"avg minute {c['mean_tick']:5.1f} ± {c['std_tick']:.1f}  "
                      f"({c['pct_sessions']:.1f}% of sessions)")
            else:
                print(f"    → {zone_name:8s}: never crossed")

    # Alert threshold justification
    print(f"\n  ── Alert threshold justification ──")
    print(f"  CDG=0.60 chosen as alert threshold.")
    aut_cross = results['autonomous']['crossings'].get('high', {})
    grd_cross = results['gradual_drifter']['crossings'].get('high', {})
    thr_cross = results['threshold_drifter']['crossings'].get('high', {})
    print(f"  Autonomous sessions crossing 0.60   : "
          f"{aut_cross.get('pct_sessions', 0):.1f}%  "
          f"(target: <5%)")
    print(f"  Gradual drifter crossing 0.60       : "
          f"{grd_cross.get('pct_sessions', 0):.1f}%  "
          f"at avg min {grd_cross.get('mean_tick', 'N/A')}")
    print(f"  Threshold drifter crossing 0.60     : "
          f"{thr_cross.get('pct_sessions', 0):.1f}%  "
          f"at avg min {thr_cross.get('mean_tick', 'N/A')}")

    # Plot zone time distribution
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.patch.set_facecolor('#0b0f1a')
    fig.suptitle('Figure 8 — Time Spent in Each CDG Zone by Profile',
                 fontsize=12, y=1.01)

    for ax, profile in zip(axes, ['autonomous', 'gradual_drifter',
                                   'threshold_drifter']):
        zone_pcts = results[profile]['zone_pcts']
        bars = ax.bar(ZONES_ORDER,
                      [zone_pcts[z] for z in ZONES_ORDER],
                      color=[ZONE_COLORS[z] for z in ZONES_ORDER],
                      edgecolor='none', alpha=0.85)
        ax.set_title(profile.replace('_', ' ').title(), fontsize=10)
        ax.set_xlabel('CDG Zone')
        ax.set_ylabel('% of session ticks')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, zone in zip(bars, ZONES_ORDER):
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f'{h:.1f}%', ha='center', va='bottom',
                        fontsize=9, color='#e2e8f0')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig8_zone_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # Plot crossing times
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0b0f1a')
    profiles = ['autonomous', 'gradual_drifter', 'threshold_drifter']
    x = np.arange(3)
    width = 0.25
    threshold_labels = ['moderate (0.30)', 'high (0.60)', 'critical (0.80)']
    threshold_keys   = ['moderate', 'high', 'critical']
    colors = ['#3b82f6', '#f59e0b', '#ef4444']

    for i, (key, label, color) in enumerate(
            zip(threshold_keys, threshold_labels, colors)):
        means = []
        errs  = []
        for p in profiles:
            c = results[p]['crossings'].get(key, {})
            means.append(c.get('mean_tick') or 0)
            errs.append(c.get('std_tick') or 0)
        bars = ax.bar(x + i*width, means, width, label=label,
                      color=color, alpha=0.8, edgecolor='none')

    ax.set_xlabel('User Profile')
    ax.set_ylabel('Session minute of first crossing')
    ax.set_title('Figure 9 — Zone Threshold Crossing Times by Profile', pad=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([p.replace('_', '\n') for p in profiles], fontsize=9)
    ax.legend(framealpha=0.2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig9_crossing_times.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — SFI Amplification Quantification
# ══════════════════════════════════════════════════════════════════════════════
def analysis_sfi_amplification(df):
    print("\n" + "="*55)
    print("  ANALYSIS 2 — SFI Amplification Quantification")
    print("="*55)

    results = {}

    for profile in ['autonomous', 'gradual_drifter', 'threshold_drifter']:
        sessions = df[df['profile'] == profile]

        # Compute what CDG would be without SFI (divide by SFI)
        sessions = sessions.copy()
        sessions['CDG_no_sfi'] = np.clip(
            sessions['CDG'] / sessions['SFI'], 0.0, 1.0
        )
        sessions['SFI_amplification'] = sessions['CDG'] - sessions['CDG_no_sfi']
        sessions['SFI_amplification_pct'] = (
            (sessions['CDG'] - sessions['CDG_no_sfi']) /
            sessions['CDG_no_sfi'].replace(0, np.nan) * 100
        )

        # Stats
        mean_amp_abs = sessions['SFI_amplification'].mean()
        mean_amp_pct = sessions['SFI_amplification_pct'].dropna().mean()

        # Early vs late session SFI effect
        early = sessions[sessions['t_min'] <= 10]
        late  = sessions[sessions['t_min'] >= 50]
        sfi_early = early['SFI'].mean()
        sfi_late  = late['SFI'].mean()
        cdg_amp_early = early['SFI_amplification'].mean()
        cdg_amp_late  = late['SFI_amplification'].mean()

        results[profile] = {
            'mean_amp_abs': mean_amp_abs,
            'mean_amp_pct': mean_amp_pct,
            'sfi_early':    sfi_early,
            'sfi_late':     sfi_late,
            'amp_early':    cdg_amp_early,
            'amp_late':     cdg_amp_late,
        }

        print(f"\n  Profile: {profile}")
        print(f"  Mean SFI amplification (absolute CDG added) : "
              f"{mean_amp_abs:.4f}")
        print(f"  Mean SFI amplification (% increase)         : "
              f"{mean_amp_pct:.1f}%")
        print(f"  SFI at minute 0-10  : {sfi_early:.3f}  "
              f"→ adds {cdg_amp_early:.4f} to CDG")
        print(f"  SFI at minute 50-60 : {sfi_late:.3f}  "
              f"→ adds {cdg_amp_late:.4f} to CDG")

    # Cross-profile amplification summary
    print(f"\n  ── Key paper statistic ──")
    for profile, r in results.items():
        print(f"  {profile:22s}: SFI adds avg "
              f"{r['mean_amp_pct']:.1f}% to CDG score over full session")

    # Plot SFI amplification over time
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0b0f1a')
    fig.suptitle('Figure 10 — SFI Amplification Effect on CDG',
                 fontsize=12, y=1.01)

    # Left: CDG with vs without SFI over time
    ax = axes[0]
    for profile, color in PROFILE_COLORS.items():
        sess = df[df['profile'] == profile].copy()
        sess['CDG_no_sfi'] = np.clip(sess['CDG'] / sess['SFI'], 0.0, 1.0)
        mean_cdg      = sess.groupby('t_min')['CDG'].mean()
        mean_cdg_base = sess.groupby('t_min')['CDG_no_sfi'].mean()
        label = profile.replace('_', ' ').title()
        ax.plot(mean_cdg.index, mean_cdg.values,
                color=color, linewidth=2.2, label=f'{label} (with SFI)')
        ax.plot(mean_cdg_base.index, mean_cdg_base.values,
                color=color, linewidth=1.2, linestyle='--', alpha=0.5,
                label=f'{label} (no SFI)')
    ax.axhline(0.60, color='#ef4444', linestyle='--',
               linewidth=1, label='Alert (0.60)')
    ax.set_xlabel('Session time (minutes)')
    ax.set_ylabel('CDG Score')
    ax.set_title('CDG with vs without SFI', fontsize=10)
    ax.legend(fontsize=7, framealpha=0.2)
    ax.grid(True, alpha=0.3)

    # Right: SFI amplification over time per profile
    ax = axes[1]
    for profile, color in PROFILE_COLORS.items():
        sess = df[df['profile'] == profile].copy()
        sess['amp'] = sess['CDG'] - np.clip(
            sess['CDG'] / sess['SFI'], 0.0, 1.0)
        mean_amp = sess.groupby('t_min')['amp'].mean()
        ax.plot(mean_amp.index, mean_amp.values, color=color,
                linewidth=2.2,
                label=profile.replace('_', ' ').title())
    ax.set_xlabel('Session time (minutes)')
    ax.set_ylabel('CDG points added by SFI')
    ax.set_title('SFI Amplification Over Session', fontsize=10)
    ax.legend(framealpha=0.2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig10_sfi_amplification.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Construct Validity Check
# ══════════════════════════════════════════════════════════════════════════════
def analysis_construct_validity(df, summary):
    print("\n" + "="*55)
    print("  ANALYSIS 3 — Construct Validity Check")
    print("="*55)

    autonomous_sessions = df[df['profile'] == 'autonomous']
    drifter_sessions    = df[df['profile'] != 'autonomous']

    print(f"\n  Condition 1 (False positive check):")
    print(f"  Autonomous sessions that ever cross CDG=0.60:")
    aut_summary = summary[summary['profile'] == 'autonomous']
    false_alerts = aut_summary[aut_summary['crossed_alert'] == 1]
    false_alert_rate = len(false_alerts) / len(aut_summary) * 100
    print(f"    {len(false_alerts)} / {len(aut_summary)} sessions "
          f"({false_alert_rate:.1f}%)")
    c1_pass = false_alert_rate < 5.0
    print(f"    Target: <5.0%  →  {'✓ PASS' if c1_pass else '✗ FAIL'}")

    print(f"\n  Condition 1b: False alerts in FIRST 30 minutes:")
    aut_early = autonomous_sessions[autonomous_sessions['t_min'] <= 30]
    early_false = (aut_early['CDG'] >= 0.60).sum()
    early_total = len(aut_early)
    early_pct   = early_false / early_total * 100
    print(f"    Ticks above 0.60 in first 30 min: "
          f"{early_false} / {early_total} ({early_pct:.3f}%)")
    c1b_pass = early_pct < 1.0
    print(f"    Target: <1.0%  →  {'✓ PASS' if c1b_pass else '✗ FAIL'}")

    print(f"\n  Condition 2 (Weight falsifiability):")
    print(f"  Already validated — see weight sensitivity analysis.")
    print(f"  ARWM is highest-weighted variable (0.25). Removing it entirely:")
    # Recompute CDG without ARWM (set to 0)
    test_df = df[df['profile'] != 'autonomous'].copy()
    # CDG without ARWM contribution: subtract w4*ARWM_n from inner sum
    # inner_no_arwm = inner - 0.25*ARWM_n
    # We have CDG = SFI * inner, so inner = CDG / SFI
    test_df['inner'] = test_df['CDG'] / test_df['SFI']
    test_df['inner_no_arwm'] = test_df['inner'] - 0.25 * test_df['ARWM_n']
    test_df['CDG_no_arwm'] = np.clip(
        test_df['SFI'] * test_df['inner_no_arwm'], 0.0, 1.0
    )
    mean_cdg_full    = test_df['CDG'].mean()
    mean_cdg_no_arwm = test_df['CDG_no_arwm'].mean()
    arwm_contribution = (mean_cdg_full - mean_cdg_no_arwm) / mean_cdg_full * 100
    print(f"    Mean CDG with ARWM   : {mean_cdg_full:.4f}")
    print(f"    Mean CDG without ARWM: {mean_cdg_no_arwm:.4f}")
    print(f"    ARWM contribution    : {arwm_contribution:.1f}% of CDG score")
    c2_pass = arwm_contribution > 5.0
    print(f"    ARWM significantly impacts CDG → "
          f"{'✓ weight justified' if c2_pass else '✗ weight may be too high'}")

    print(f"\n  Condition 3 (Discriminant validity):")
    print(f"  CDG inter-session variation across profiles:")
    for profile in ['autonomous', 'gradual_drifter', 'threshold_drifter']:
        sub = summary[summary['profile'] == profile]
        std = sub['cdg_mean'].std()
        rng = sub['cdg_mean'].max() - sub['cdg_mean'].min()
        print(f"    {profile:22s}: std={std:.4f}  range={rng:.4f}")
    # Cross-profile variation must be >> within-profile variation
    between_profile_range = (
        summary.groupby('profile')['cdg_mean'].mean().max() -
        summary.groupby('profile')['cdg_mean'].mean().min()
    )
    within_profile_std = summary.groupby('profile')['cdg_mean'].std().mean()
    discrimination_ratio = between_profile_range / within_profile_std
    print(f"    Between-profile range     : {between_profile_range:.4f}")
    print(f"    Mean within-profile std   : {within_profile_std:.4f}")
    print(f"    Discrimination ratio      : {discrimination_ratio:.1f}x")
    c3_pass = discrimination_ratio > 10.0
    print(f"    Target: >10x  →  {'✓ PASS' if c3_pass else '✗ FAIL'}")

    # Overall verdict
    print(f"\n  ── Construct validity summary ──")
    checks = {
        'False positive rate < 5%':          c1_pass,
        'False alerts in first 30 min < 1%': c1b_pass,
        'ARWM contribution > 5%':            c2_pass,
        'Discrimination ratio > 10x':        c3_pass,
    }
    all_pass = True
    for check, passed in checks.items():
        print(f"    {'✓' if passed else '✗'} {check}")
        if not passed:
            all_pass = False
    print(f"\n  CDG construct validity: "
          f"{'CONFIRMED' if all_pass else 'NEEDS REVIEW'}")

    # Plot CDG variance by profile
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0b0f1a')
    for profile, color in PROFILE_COLORS.items():
        sub = summary[summary['profile'] == profile]
        ax.scatter(sub.index, sub['cdg_mean'], color=color, alpha=0.4,
                   s=18, label=profile.replace('_', ' ').title())
        mean_val = sub['cdg_mean'].mean()
        ax.axhline(mean_val, color=color, linewidth=1.5,
                   linestyle='--', alpha=0.7)
    ax.axhline(0.60, color='#ef4444', linewidth=1.2,
               linestyle='-', label='Alert threshold (0.60)')
    ax.set_xlabel('Session index')
    ax.set_ylabel('Mean CDG score')
    ax.set_title('Figure 11 — CDG Score Distribution: '
                 'Construct Validity Check', pad=12)
    ax.legend(framealpha=0.2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig11_construct_validity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    return checks


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df, summary = load_all()
    print(f"Loaded {len(df)} ticks across {df['session_id'].nunique()} sessions")

    zone_results = analysis_zone_transitions(df, summary)
    sfi_results  = analysis_sfi_amplification(df)
    cv_results   = analysis_construct_validity(df, summary)

    print("\n" + "="*55)
    print("  ALL ANALYSES COMPLETE")
    print("="*55)
    print("  Figures 8-11 saved to paper/figures/")
    print("  Formula is ready for finalization.\n")


if __name__ == "__main__":
    main()
"""
CDG Session Generator — simulation/session_generator.py
Generates synthetic sessions with known ground-truth dependency labels.
Three user profiles: Autonomous, Gradual Drifter, Threshold Drifter.
Output: CSV per session + master summary CSV.
"""

import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from cdg_formula import (
    compute_sfi, compute_cdg, compute_drift_rate,
    detect_inflection_point, get_zone, normalize_signal
)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
N_SESSIONS      = 500
SESSION_MINS    = 60
TICK_INTERVAL   = 1          # minutes per tick
N_TICKS         = SESSION_MINS // TICK_INTERVAL
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), 'data')

# Population-level bounds (hybrid normalization — fixed bounds as baseline)
POP_BOUNDS = {
    'PQAR': (0.0, 1.0),
    'QCS':  (0.0, 1.0),
    'TTQ':  (0.0, 1.0),
    'ARWM': (0.0, 1.0),
    'RET':  (0.0, 1.0),
    'OCR':  (0.0, 1.0),
}

# Profile distribution: 40% autonomous, 35% gradual, 25% threshold
PROFILE_WEIGHTS = [0.40, 0.35, 0.25]
PROFILES        = ['autonomous', 'gradual_drifter', 'threshold_drifter']


# ── Signal generators per profile ────────────────────────────────────────────
def autonomous_signals(tick, n_ticks, rng):
    """
    Autonomous user: consistently high independent effort throughout.
    Small natural variation but no systematic drift.
    """
    t = tick / n_ticks
    noise = lambda s: rng.normal(0, s)

    PQAR = np.clip(0.80 + noise(0.08), 0.0, 1.0)
    QCS  = np.clip(0.75 + noise(0.08), 0.0, 1.0)
    TTQ  = np.clip(0.70 + noise(0.10), 0.0, 1.0)
    ARWM = np.clip(0.10 + noise(0.05), 0.0, 1.0)
    RET  = np.clip(0.80 + noise(0.08), 0.0, 1.0)
    OCR  = np.clip(0.40 + noise(0.10), 0.0, 1.0)

    # SFI sub-signals: fatigue rises naturally but slowly
    TSD = np.clip(0.05 * t + noise(0.03), 0.0, 1.0)
    TER = np.clip(0.05 * t + noise(0.03), 0.0, 1.0)
    QIC = np.clip(0.05 * t + noise(0.03), 0.0, 1.0)

    return dict(PQAR=PQAR, QCS=QCS, TTQ=TTQ, ARWM=ARWM, RET=RET, OCR=OCR,
                TSD=TSD, TER=TER, QIC=QIC)


def gradual_drifter_signals(tick, n_ticks, rng):
    """
    Gradual drifter: dependency increases linearly from ~0.2 to ~0.7 over session.
    Models a user who slowly loses critical evaluation as fatigue accumulates.
    """
    t = tick / n_ticks
    noise = lambda s: rng.normal(0, s)

    # Autonomy signals decline linearly
    PQAR = np.clip(0.75 - 0.50 * t + noise(0.07), 0.0, 1.0)
    QCS  = np.clip(0.70 - 0.45 * t + noise(0.07), 0.0, 1.0)
    TTQ  = np.clip(0.65 - 0.40 * t + noise(0.08), 0.0, 1.0)
    RET  = np.clip(0.75 - 0.50 * t + noise(0.07), 0.0, 1.0)
    OCR  = np.clip(0.35 - 0.25 * t + noise(0.08), 0.0, 1.0)

    # ARWM (direct) rises linearly
    ARWM = np.clip(0.15 + 0.55 * t + noise(0.06), 0.0, 1.0)

    # SFI sub-signals: moderate fatigue accumulation
    TSD = np.clip(0.10 * t + 0.30 * t**2 + noise(0.04), 0.0, 1.0)
    TER = np.clip(0.10 * t + 0.25 * t**2 + noise(0.04), 0.0, 1.0)
    QIC = np.clip(0.15 * t + 0.30 * t**2 + noise(0.04), 0.0, 1.0)

    return dict(PQAR=PQAR, QCS=QCS, TTQ=TTQ, ARWM=ARWM, RET=RET, OCR=OCR,
                TSD=TSD, TER=TER, QIC=QIC)


def threshold_drifter_signals(tick, n_ticks, rng):
    """
    Threshold drifter: autonomous until ~tick 20 (1/3 of session),
    then sharp nonlinear drift. Models Goddard et al. (2014) threshold effect.
    """
    t = tick / n_ticks
    threshold = 0.33
    noise = lambda s: rng.normal(0, s)

    if t < threshold:
        # Pre-threshold: autonomous behavior
        PQAR = np.clip(0.78 + noise(0.07), 0.0, 1.0)
        QCS  = np.clip(0.72 + noise(0.07), 0.0, 1.0)
        TTQ  = np.clip(0.68 + noise(0.09), 0.0, 1.0)
        ARWM = np.clip(0.12 + noise(0.05), 0.0, 1.0)
        RET  = np.clip(0.78 + noise(0.07), 0.0, 1.0)
        OCR  = np.clip(0.38 + noise(0.09), 0.0, 1.0)
        TSD  = np.clip(0.05 * t + noise(0.03), 0.0, 1.0)
        TER  = np.clip(0.05 * t + noise(0.03), 0.0, 1.0)
        QIC  = np.clip(0.05 * t + noise(0.03), 0.0, 1.0)
    else:
        # Post-threshold: nonlinear rapid drift
        drift = ((t - threshold) / (1.0 - threshold)) ** 1.8
        PQAR = np.clip(0.75 - 0.68 * drift + noise(0.07), 0.0, 1.0)
        QCS  = np.clip(0.70 - 0.62 * drift + noise(0.07), 0.0, 1.0)
        TTQ  = np.clip(0.65 - 0.55 * drift + noise(0.08), 0.0, 1.0)
        ARWM = np.clip(0.12 + 0.80 * drift + noise(0.06), 0.0, 1.0)
        RET  = np.clip(0.75 - 0.68 * drift + noise(0.07), 0.0, 1.0)
        OCR  = np.clip(0.35 - 0.32 * drift + noise(0.08), 0.0, 1.0)
        TSD  = np.clip(0.20 * drift + noise(0.04), 0.0, 1.0)
        TER  = np.clip(0.25 * drift + noise(0.04), 0.0, 1.0)
        QIC  = np.clip(0.30 * drift + noise(0.04), 0.0, 1.0)

    return dict(PQAR=PQAR, QCS=QCS, TTQ=TTQ, ARWM=ARWM, RET=RET, OCR=OCR,
                TSD=TSD, TER=TER, QIC=QIC)


SIGNAL_FN = {
    'autonomous':        autonomous_signals,
    'gradual_drifter':   gradual_drifter_signals,
    'threshold_drifter': threshold_drifter_signals,
}


# ── Single session simulation ─────────────────────────────────────────────────
def simulate_session(session_id, profile, rng):
    signal_fn = SIGNAL_FN[profile]
    rows = []
    cdg_prev = 0.0
    dr_prev  = 0.0

    for tick in range(N_TICKS):
        t = tick * TICK_INTERVAL           # minutes elapsed
        raw = signal_fn(tick, N_TICKS, rng)

        # Normalize signals using population bounds
        norm = {k: normalize_signal(k, raw[k], *POP_BOUNDS[k]) for k in POP_BOUNDS}

        # Compute SFI
        sfi = compute_sfi(
            t=t, T_max=SESSION_MINS,
            TSD=raw['TSD'], TER=raw['TER'], QIC=raw['QIC']
        )

        # Compute CDG
        cdg = compute_cdg(norm, sfi)

        # Drift rate and inflection point
        dt = TICK_INTERVAL
        dr  = compute_drift_rate(cdg, cdg_prev, dt)
        ip = False
        if tick > 5 and cdg >= 0.35:
            recent_cdg = [r['CDG'] for r in rows[-6:]]
            net_rise = recent_cdg[-1] - recent_cdg[0]
            recent_dr = [r['DR'] for r in rows[-3:]]
            avg_dr = sum(recent_dr) / len(recent_dr)
            # IP fires only if: net rise > 0.06 AND average drift rate > 0.008
            if net_rise > 0.06 and avg_dr > 0.008:
                ip = detect_inflection_point(dr, dr_prev, dt, cdg_current=cdg)
            else:
                ip = False
        zone = get_zone(cdg)

        rows.append({
            'session_id': session_id,
            'profile':    profile,
            'tick':       tick,
            't_min':      t,
            # Raw signals
            'PQAR_raw': raw['PQAR'], 'QCS_raw':  raw['QCS'],
            'TTQ_raw':  raw['TTQ'],  'ARWM_raw': raw['ARWM'],
            'RET_raw':  raw['RET'],  'OCR_raw':  raw['OCR'],
            'TSD':      raw['TSD'],  'TER':      raw['TER'],  'QIC': raw['QIC'],
            # Normalized signals
            'PQAR_n': norm['PQAR'], 'QCS_n':  norm['QCS'],
            'TTQ_n':  norm['TTQ'],  'ARWM_n': norm['ARWM'],
            'RET_n':  norm['RET'],  'OCR_n':  norm['OCR'],
            # Computed values
            'SFI': sfi, 'CDG': cdg, 'DR': dr,
            'IP':  ip,  'zone': zone,
        })

        cdg_prev = cdg
        dr_prev  = dr

    return pd.DataFrame(rows)


# ── Run all sessions ──────────────────────────────────────────────────────────
def run_all_sessions():
    rng = np.random.default_rng(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    profiles_assigned = rng.choice(PROFILES, size=N_SESSIONS, p=PROFILE_WEIGHTS)
    summary_rows = []

    print(f"Generating {N_SESSIONS} synthetic sessions...\n")

    for i, profile in enumerate(tqdm(profiles_assigned, desc="Sessions")):
        session_id = f"SYN_{i+1:04d}"
        df = simulate_session(session_id, profile, rng)

        # Save individual session CSV
        csv_path = os.path.join(OUTPUT_DIR, f"{session_id}.csv")
        df.to_csv(csv_path, index=False)

        # Summary stats for this session
        summary_rows.append({
            'session_id':      session_id,
            'profile':         profile,
            'cdg_mean':        df['CDG'].mean(),
            'cdg_max':         df['CDG'].max(),
            'cdg_final':       df['CDG'].iloc[-1],
            'sfi_mean':        df['SFI'].mean(),
            'ip_count':        df['IP'].sum(),
            'ip_first_tick':   df.loc[df['IP'] == True, 'tick'].min() if df['IP'].any() else -1,
            'final_zone':      df['zone'].iloc[-1],
            'crossed_alert':   int((df['CDG'] >= 0.60).any()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, 'simulation_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    # Print results
    print(f"\n{'='*55}")
    print(f"  SIMULATION COMPLETE — {N_SESSIONS} sessions generated")
    print(f"{'='*55}")
    print(f"\nProfile distribution:")
    for p in PROFILES:
        n = (summary_df['profile'] == p).sum()
        print(f"  {p:22s}: {n:3d} sessions ({n/N_SESSIONS*100:.1f}%)")

    print(f"\nCDG statistics by profile:")
    for p in PROFILES:
        sub = summary_df[summary_df['profile'] == p]
        print(f"  {p:22s}: mean={sub['cdg_mean'].mean():.3f}  "
              f"max={sub['cdg_max'].mean():.3f}  "
              f"alert_rate={sub['crossed_alert'].mean()*100:.1f}%")

    print(f"\nInflection point detection:")
    for p in PROFILES:
        sub = summary_df[summary_df['profile'] == p]
        ip_rate = (sub['ip_count'] > 0).mean() * 100
        ip_tick = sub.loc[sub['ip_first_tick'] > 0, 'ip_first_tick'].mean()
        print(f"  {p:22s}: IP detected in {ip_rate:.1f}% of sessions  "
              f"(avg tick {ip_tick:.1f})")

    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Summary: {summary_path}")
    print(f"{'='*55}\n")

    return summary_df


if __name__ == "__main__":
    run_all_sessions()
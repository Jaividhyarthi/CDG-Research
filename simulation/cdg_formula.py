"""
CDG Formula Core — simulation/cdg_formula.py
Cognitive Dependency Gradient: single source of truth for all formula logic.
Every component imports from here. Never duplicate this math elsewhere.
"""

import numpy as np

# ── Formula constants ─────────────────────────────────────────────────────────
WEIGHTS = {
    'PQAR': 0.20,   # Pre-Query Attempt Rate       (inverse)
    'QCS':  0.15,   # Query Complexity Score        (inverse)
    'TTQ':  0.10,   # Time to Query                 (inverse)
    'ARWM': 0.25,   # Acceptance Rate w/o Mod       (direct)  ← highest weight
    'RET':  0.15,   # Response Evaluation Time      (inverse)
    'OCR':  0.15,   # Override / Correction Rate    (inverse)
}

SFI_WEIGHTS = {
    't_progress': 0.40,   # Session time progress (t / T_max)
    'TSD':        0.25,   # Typing Speed Decline
    'TER':        0.15,   # Typing Error Rate increase
    'QIC':        0.20,   # Query Interval Compression
}

SFI_MIN = 1.0
SFI_MAX = 2.0

CDG_ZONES = {
    'low':      (0.0, 0.3),
    'moderate': (0.3, 0.6),
    'high':     (0.6, 0.8),
    'critical': (0.8, 1.0),
}

ALERT_THRESHOLD = 0.60   # CDG value that triggers extension warning


# ── Normalization ─────────────────────────────────────────────────────────────
def normalize_inverse(v, v_min, v_max):
    """Inverse variable: high raw value = more autonomous → flip so high = more dependent."""
    if v_max == v_min:
        return 0.0
    return 1.0 - ((v - v_min) / (v_max - v_min))


def normalize_direct(v, v_min, v_max):
    """Direct variable (ARWM only): high raw value already = more dependent."""
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


def normalize_signal(name, v, v_min, v_max):
    """Route to correct normalization based on variable direction."""
    v = np.clip(v, v_min, v_max)
    if name == 'ARWM':
        return normalize_direct(v, v_min, v_max)
    return normalize_inverse(v, v_min, v_max)


# ── Session Fatigue Index ─────────────────────────────────────────────────────
def compute_sfi(t, T_max, TSD, TER, QIC):
    """
    SFI(t) = 1 + [0.40*(t/T_max) + 0.25*TSD + 0.15*TER + 0.20*QIC]
    Range: 1.0 (fresh) → 2.0 (peak fatigue, strictly capped).
    All sub-signals must be pre-normalized to [0, 1].
    """
    t_progress = np.clip(t / T_max, 0.0, 1.0) if T_max > 0 else 0.0
    inner = (
        SFI_WEIGHTS['t_progress'] * t_progress +
        SFI_WEIGHTS['TSD']        * np.clip(TSD, 0.0, 1.0) +
        SFI_WEIGHTS['TER']        * np.clip(TER, 0.0, 1.0) +
        SFI_WEIGHTS['QIC']        * np.clip(QIC, 0.0, 1.0)
    )
    return np.clip(1.0 + inner, SFI_MIN, SFI_MAX)


# ── Primary CDG Formula ───────────────────────────────────────────────────────
def compute_cdg(signals_norm, sfi):
    """
    CDG(t) = SFI(t) × [w1·PQAR' + w2·QCS' + w3·TTQ' + w4·ARWM + w5·RET' + w6·OCR']
    signals_norm: dict of already-normalized signal values (all in [0,1]).
    sfi: Session Fatigue Index (1.0 – 2.0).
    Returns CDG in [0.0, 1.0].
    """
    inner = sum(WEIGHTS[k] * signals_norm[k] for k in WEIGHTS)
    cdg = sfi * inner
    return float(np.clip(cdg, 0.0, 1.0))


# ── Drift Rate ────────────────────────────────────────────────────────────────
def compute_drift_rate(cdg_current, cdg_prev, dt):
    """DR(t) = [CDG(t) - CDG(t-dt)] / dt. Positive = drifting, negative = recovering."""
    if dt <= 0:
        return 0.0
    return (cdg_current - cdg_prev) / dt


# ── Inflection Point ──────────────────────────────────────────────────────────
def detect_inflection_point(dr_current, dr_prev, dt):
    """
    IP when d²CDG/dt² changes sign from negative to positive.
    Returns True if inflection point detected at this tick.
    """
    if dt <= 0:
        return False
    d2 = (dr_current - dr_prev) / dt
    return dr_prev <= 0 < d2


# ── Zone classification ───────────────────────────────────────────────────────
def get_zone(cdg):
    for zone, (lo, hi) in CDG_ZONES.items():
        if lo <= cdg <= hi:
            return zone
    return 'critical'


# ── Unit tests ────────────────────────────────────────────────────────────────
def run_tests():
    print("Running CDG formula unit tests...\n")
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name} FAILED {detail}")
            failed += 1

    # Test 1: Zero dependency → CDG = 0
    # All signals at normalized 0.0 = fully autonomous
    sigs = {'PQAR': 0.0, 'QCS': 0.0, 'TTQ': 0.0, 'ARWM': 0.0, 'RET': 0.0, 'OCR': 0.0}
    cdg = compute_cdg(sigs, sfi=1.0)
    check("Zero dependency → CDG = 0.0", cdg == 0.0, f"got {cdg}")

    # Test 2: Full dependency → CDG = 1.0 (with max SFI=2, all signals=0.5 → CDG=1.0)
    sigs = {'PQAR': 0.5, 'QCS': 0.5, 'TTQ': 0.5, 'ARWM': 0.5, 'RET': 0.5, 'OCR': 0.5}
    cdg = compute_cdg(sigs, sfi=2.0)
    check("All signals at 0.5 + SFI=2.0 → CDG = 1.0", abs(cdg - 1.0) < 1e-9, f"got {cdg}")
    
    # Test 3: CDG always bounded [0, 1]
    for _ in range(1000):
        sigs = {k: np.random.random() for k in WEIGHTS}
        sfi = np.random.uniform(1.0, 2.0)
        cdg = compute_cdg(sigs, sfi)
        if not (0.0 <= cdg <= 1.0):
            check("CDG bounded [0,1] random test", False, f"got {cdg}")
            break
    else:
        check("CDG bounded [0,1] across 1000 random inputs", True)

    # Test 4: SFI strictly capped at 2.0
    sfi = compute_sfi(t=9999, T_max=60, TSD=1.0, TER=1.0, QIC=1.0)
    check("SFI capped at 2.0 under extreme fatigue", sfi == 2.0, f"got {sfi}")

    # Test 5: SFI = 1.0 at session start
    sfi = compute_sfi(t=0, T_max=60, TSD=0.0, TER=0.0, QIC=0.0)
    check("SFI = 1.0 at fresh start", sfi == 1.0, f"got {sfi}")

    # Test 6: Weights sum to 1.0
    check("Weights sum to 1.0", abs(sum(WEIGHTS.values()) - 1.0) < 1e-9)

    # Test 7: SFI weights sum to 1.0
    check("SFI weights sum to 1.0", abs(sum(SFI_WEIGHTS.values()) - 1.0) < 1e-9)

    # Test 8: Inverse normalization direction correct
    v_norm = normalize_inverse(v=0.9, v_min=0.0, v_max=1.0)
    check("Inverse norm: high autonomy → low dependency score", v_norm < 0.2, f"got {v_norm}")

    # Test 9: Direct normalization direction correct
    v_norm = normalize_direct(v=0.9, v_min=0.0, v_max=1.0)
    check("Direct norm: high ARWM → high dependency score", v_norm > 0.8, f"got {v_norm}")

    # Test 10: Inflection point detection
    ip = detect_inflection_point(dr_current=0.01, dr_prev=-0.005, dt=1.0)
    check("IP detected when DR crosses from negative to positive", ip)

    # Test 11: No IP when DR stays positive
    ip = detect_inflection_point(dr_current=0.02, dr_prev=0.01, dt=1.0)
    check("No IP when DR stays positive throughout", not ip)

    # Test 12: Zone classification
    check("CDG=0.15 → zone low",      get_zone(0.15) == 'low')
    check("CDG=0.45 → zone moderate", get_zone(0.45) == 'moderate')
    check("CDG=0.70 → zone high",     get_zone(0.70) == 'high')
    check("CDG=0.90 → zone critical", get_zone(0.90) == 'critical')

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed. CDG formula core is verified.")
    else:
        print("Fix failing tests before proceeding.")
    return failed == 0


if __name__ == "__main__":
    run_tests()
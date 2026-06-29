/**
 * CDG Engine — extension/src/cdg_engine.js
 * CDG formula core translated from Python cdg_formula.py
 * Single source of truth for all formula logic in the extension.
 * Imported by background.js (service worker).
 */

// ── Formula constants ──────────────────────────────────────────────────────
export const WEIGHTS = {
  PQAR: 0.20,
  QCS:  0.15,
  TTQ:  0.10,
  ARWM: 0.25,
  RET:  0.15,
  OCR:  0.15,
};

export const SFI_WEIGHTS = {
  t_progress: 0.40,
  TSD:        0.25,
  TER:        0.15,
  QIC:        0.20,
};

export const SFI_MIN = 1.0;
export const SFI_MAX = 2.0;
export const ALERT_THRESHOLD = 0.60;
export const MIN_CDG_FOR_IP  = 0.35;

export const CDG_ZONES = {
  low:      [0.0, 0.3],
  moderate: [0.3, 0.6],
  high:     [0.6, 0.8],
  critical: [0.8, 1.0],
};

// ── Normalization ──────────────────────────────────────────────────────────
export function normalizeInverse(v, vMin, vMax) {
  if (vMax === vMin) return 0.0;
  const clamped = Math.max(vMin, Math.min(vMax, v));
  return 1.0 - ((clamped - vMin) / (vMax - vMin));
}

export function normalizeDirect(v, vMin, vMax) {
  if (vMax === vMin) return 0.0;
  const clamped = Math.max(vMin, Math.min(vMax, v));
  return (clamped - vMin) / (vMax - vMin);
}

export function normalizeSignal(name, v, vMin, vMax) {
  return name === 'ARWM'
    ? normalizeDirect(v, vMin, vMax)
    : normalizeInverse(v, vMin, vMax);
}

// ── Session Fatigue Index ──────────────────────────────────────────────────
export function computeSFI(t, tMax, TSD, TER, QIC) {
  const tProgress = tMax > 0 ? Math.min(1.0, t / tMax) : 0.0;
  const inner = (
    SFI_WEIGHTS.t_progress * tProgress +
    SFI_WEIGHTS.TSD        * Math.min(1.0, Math.max(0.0, TSD)) +
    SFI_WEIGHTS.TER        * Math.min(1.0, Math.max(0.0, TER)) +
    SFI_WEIGHTS.QIC        * Math.min(1.0, Math.max(0.0, QIC))
  );
  return Math.min(SFI_MAX, Math.max(SFI_MIN, 1.0 + inner));
}

// ── Primary CDG formula ────────────────────────────────────────────────────
export function computeCDG(signalsNorm, sfi) {
  const inner = Object.entries(WEIGHTS)
    .reduce((sum, [k, w]) => sum + w * (signalsNorm[k] || 0.0), 0.0);
  return Math.min(1.0, Math.max(0.0, sfi * inner));
}

// ── Drift rate ─────────────────────────────────────────────────────────────
export function computeDriftRate(cdgCurrent, cdgPrev, dt) {
  if (dt <= 0) return 0.0;
  return (cdgCurrent - cdgPrev) / dt;
}

// ── Inflection point ───────────────────────────────────────────────────────
export function detectInflectionPoint(drCurrent, drPrev, dt, cdgCurrent, recentCDG) {
  if (dt <= 0 || cdgCurrent < MIN_CDG_FOR_IP) return false;
  if (recentCDG && recentCDG.length >= 5) {
    const netRise = recentCDG[recentCDG.length - 1] - recentCDG[0];
    if (netRise <= 0.08) return false;
  }
  const d2 = (drCurrent - drPrev) / dt;
  return drPrev <= 0 && d2 > 0;
}

// ── Zone classification ────────────────────────────────────────────────────
export function getZone(cdg) {
  for (const [zone, [lo, hi]] of Object.entries(CDG_ZONES)) {
    if (cdg >= lo && cdg <= hi) return zone;
  }
  return 'critical';
}

// ── Population bounds (hybrid normalization fallback) ─────────────────────
export const POPULATION_BOUNDS = {
  PQAR: { min: 0.0, max: 1.0 },
  QCS:  { min: 0.0, max: 1.0 },
  TTQ:  { min: 0.0, max: 1.0 },
  ARWM: { min: 0.0, max: 1.0 },
  RET:  { min: 0.0, max: 1.0 },
  OCR:  { min: 0.0, max: 1.0 },
};
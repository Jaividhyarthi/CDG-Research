/**
 * CDG background.js — Service Worker
 * 60-second computation engine.
 * Reads signals from content.js → computes CDG → detects IP → fires alerts.
 */

import {
  computeSFI, computeCDG, computeDriftRate,
  detectInflectionPoint, getZone, normalizeSignal,
  ALERT_THRESHOLD, POPULATION_BOUNDS, WEIGHTS
} from './cdg_engine.js';

// ── Session state ──────────────────────────────────────────────────────────
let session = {
  startTime:    Date.now(),
  tick:         0,
  cdgHistory:   [],
  sfiHistory:   [],
  drHistory:    [],
  zoneHistory:  [],
  ipEvents:     [],
  alertEvents:  [],
  signalHistory:{},
  cdgPrev:      0.0,
  drPrev:       0.0,
  calibration:  null, // per-user bounds set after 10 min
  calibrationWindow: [], // raw signals during first 10 min
  CALIBRATION_TICKS: 10,
};

// ── Start alarm ────────────────────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  chrome.alarms.create('cdg-tick', { periodInMinutes: 1/60 }); // every 60s
  console.log('[CDG] Extension installed. Alarm started.');
});

chrome.runtime.onStartup.addListener(() => {
  chrome.alarms.create('cdg-tick', { periodInMinutes: 1/60 });
});

// ── Main computation tick ──────────────────────────────────────────────────
chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name !== 'cdg-tick') return;

  // Get signals from content script
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tabs.length) return;

  let signals = null;
  try {
    signals = await chrome.tabs.sendMessage(tabs[0].id, { type: 'GET_SIGNALS' });
  } catch (e) {
    return; // tab not on a supported platform
  }
  if (!signals) return;

  // ── Calibration ────────────────────────────────────────────────────────
  if (session.tick < session.CALIBRATION_TICKS) {
    session.calibrationWindow.push(signals);
    if (session.tick === session.CALIBRATION_TICKS - 1) {
      session.calibration = buildCalibration(session.calibrationWindow);
      console.log('[CDG] Per-user calibration complete:', session.calibration);
    }
  }

  // Get bounds — per-user if calibrated, population fallback otherwise
  const bounds = session.calibration || POPULATION_BOUNDS;

  // ── Normalize signals ──────────────────────────────────────────────────
  const norm = {};
  for (const k of Object.keys(WEIGHTS)) {
    const b = bounds[k] || POPULATION_BOUNDS[k];
    norm[k] = normalizeSignal(k, signals[k], b.min, b.max);
  }

  // ── Compute SFI ────────────────────────────────────────────────────────
  const t    = session.tick * 1; // 1 minute per tick
  const tMax = 60;
  const sfi  = computeSFI(t, tMax, signals.TSD, signals.TER, signals.QIC);

  // ── Compute CDG ────────────────────────────────────────────────────────
  const cdg = computeCDG(norm, sfi);

  // ── Drift rate ─────────────────────────────────────────────────────────
  const dr = computeDriftRate(cdg, session.cdgPrev, 1);

  // ── Inflection point ───────────────────────────────────────────────────
  const recentCDG = session.cdgHistory.slice(-5);
  const ip = session.tick > 5
    ? detectInflectionPoint(dr, session.drPrev, 1, cdg, recentCDG)
    : false;

  const zone = getZone(cdg);

  // ── Update session state ───────────────────────────────────────────────
  session.cdgHistory.push(cdg);
  session.sfiHistory.push(sfi);
  session.drHistory.push(dr);
  session.zoneHistory.push(zone);
  session.cdgPrev = cdg;
  session.drPrev  = dr;
  session.tick++;

  if (ip) {
    session.ipEvents.push(session.tick);
    fireAlert('INFLECTION_POINT', cdg, zone);
  }

  if (cdg >= ALERT_THRESHOLD) {
    const lastAlert = session.alertEvents[session.alertEvents.length - 1];
    if (!lastAlert || session.tick - lastAlert > 10) {
      session.alertEvents.push(session.tick);
      fireAlert('THRESHOLD_CROSSED', cdg, zone);
    }
  }

  // ── Store for popup ────────────────────────────────────────────────────
  await chrome.storage.local.set({
    cdgCurrent:   cdg,
    sfiCurrent:   sfi,
    drCurrent:    dr,
    zoneCurrent:  zone,
    cdgHistory:   session.cdgHistory,
    sfiHistory:   session.sfiHistory,
    drHistory:    session.drHistory,
    zoneHistory:  session.zoneHistory,
    ipEvents:     session.ipEvents,
    alertEvents:  session.alertEvents,
    tick:         session.tick,
    signalsNorm:  norm,
    signalsRaw:   signals,
    sessionStart: session.startTime,
  });
});

// ── Alert system ───────────────────────────────────────────────────────────
function fireAlert(type, cdg, zone) {
  const messages = {
    INFLECTION_POINT: {
      title: '⚡ CDG — Inflection Point Detected',
      message: `Dependency drift is accelerating. CDG = ${cdg.toFixed(3)} [${zone.toUpperCase()}]. Consider taking a break.`,
    },
    THRESHOLD_CROSSED: {
      title: '🔔 CDG Alert — High Dependency Detected',
      message: `CDG has crossed the alert threshold. CDG = ${cdg.toFixed(3)} [${zone.toUpperCase()}]. Review your last few AI responses critically.`,
    },
  };
  const msg = messages[type];
  if (!msg) return;

  chrome.notifications.create({
    type:    'basic',
    iconUrl: '../assets/icon48.png',
    title:   msg.title,
    message: msg.message,
    priority: 2,
  });
}

// ── Per-user calibration builder ───────────────────────────────────────────
function buildCalibration(windowSignals) {
  const keys = Object.keys(WEIGHTS);
  const bounds = {};
  for (const k of keys) {
    const vals = windowSignals.map(s => s[k]).filter(v => v !== undefined);
    if (vals.length === 0) {
      bounds[k] = POPULATION_BOUNDS[k];
    } else {
      bounds[k] = {
        min: Math.min(...vals),
        max: Math.max(...vals),
      };
      // Fallback to population bounds if range is too narrow
      if (bounds[k].max - bounds[k].min < 0.05) {
        bounds[k] = POPULATION_BOUNDS[k];
      }
    }
  }
  return bounds;
}

// ── Session reset ──────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, respond) => {
  if (msg.type === 'RESET_SESSION') {
    session = {
      startTime:    Date.now(),
      tick:         0,
      cdgHistory:   [],
      sfiHistory:   [],
      drHistory:    [],
      zoneHistory:  [],
      ipEvents:     [],
      alertEvents:  [],
      signalHistory:{},
      cdgPrev:      0.0,
      drPrev:       0.0,
      calibration:  null,
      calibrationWindow: [],
      CALIBRATION_TICKS: 10,
    };
    chrome.storage.local.clear();
    respond({ status: 'reset' });
  }

  if (msg.type === 'CDG_SIGNALS') {
    // Signals pushed proactively by content.js every 60s
    // Already handled by alarm — this is a backup
  }
});

console.log('[CDG] Background service worker loaded.');
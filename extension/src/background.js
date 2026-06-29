import {
  computeSFI, computeCDG, computeDriftRate,
  detectInflectionPoint, getZone, normalizeSignal,
  ALERT_THRESHOLD, POPULATION_BOUNDS, WEIGHTS
} from './cdg_engine.js';

let session = {
  startTime: Date.now(), tick: 0,
  cdgHistory: [], sfiHistory: [], drHistory: [], zoneHistory: [],
  ipEvents: [], alertEvents: [],
  cdgPrev: 0.0, drPrev: 0.0,
  calibration: null, calibrationWindow: [], CALIBRATION_TICKS: 10,
};

function ensureAlarm() {
  chrome.alarms.get('cdg-tick', (alarm) => {
    if (!alarm) {
      chrome.alarms.create('cdg-tick', { periodInMinutes: 1 });
      console.log('[CDG] Alarm created.');
    }
  });
}

function injectContentScript() {
  chrome.tabs.query({ url: ["https://claude.ai/*", "https://chatgpt.com/*", "https://gemini.google.com/*"] }, (tabs) => {
    tabs.forEach(tab => {
      chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['src/content.js'] }).catch(() => {});
    });
  });
}

ensureAlarm();
injectContentScript();
chrome.runtime.onInstalled.addListener(() => { ensureAlarm(); injectContentScript(); });
chrome.runtime.onStartup.addListener(() => { ensureAlarm(); injectContentScript(); });

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name !== 'cdg-tick') return;
  console.log('[CDG] Tick fired:', session.tick);

  const tabs = await chrome.tabs.query({ url: ["https://claude.ai/*", "https://chatgpt.com/*", "https://gemini.google.com/*"] });
  if (!tabs.length) { console.log('[CDG] No claude.ai tab found.'); return; }

  let signals = null;
  try {
    signals = await chrome.tabs.sendMessage(tabs[0].id, { type: 'GET_SIGNALS' });
  } catch (e) {
    console.log('[CDG] Content script unreachable:', e.message);
    injectContentScript();
    return;
  }
  if (!signals) { console.log('[CDG] No signals.'); return; }

  console.log('[CDG] Signals OK. PQAR=' + (signals.PQAR||0).toFixed(2) + ' ARWM=' + (signals.ARWM||0).toFixed(2));

  if (session.tick < session.CALIBRATION_TICKS) {
    session.calibrationWindow.push(signals);
    if (session.tick === session.CALIBRATION_TICKS - 1) {
      session.calibration = buildCalibration(session.calibrationWindow);
      console.log('[CDG] Calibration complete.');
    }
  }

  const bounds = session.calibration || POPULATION_BOUNDS;
  const norm = {};
  for (const k of Object.keys(WEIGHTS)) {
    const b = bounds[k] || POPULATION_BOUNDS[k];
    norm[k] = normalizeSignal(k, signals[k] || 0.5, b.min, b.max);
  }

  const sfi  = computeSFI(session.tick, 60, signals.TSD || 0, signals.TER || 0, signals.QIC || 0);
  const cdg  = computeCDG(norm, sfi);
  const dr   = computeDriftRate(cdg, session.cdgPrev, 1);
  const ip   = session.tick > 5 ? detectInflectionPoint(dr, session.drPrev, 1, cdg, session.cdgHistory.slice(-5)) : false;
  const zone = getZone(cdg);

  session.cdgHistory.push(cdg);
  session.sfiHistory.push(sfi);
  session.drHistory.push(dr);
  session.zoneHistory.push(zone);
  session.cdgPrev = cdg;
  session.drPrev  = dr;
  session.tick++;

  if (ip) { session.ipEvents.push(session.tick); fireAlert('INFLECTION_POINT', cdg, zone); }
  if (cdg >= ALERT_THRESHOLD) {
    const last = session.alertEvents[session.alertEvents.length - 1];
    if (!last || session.tick - last > 10) { session.alertEvents.push(session.tick); fireAlert('THRESHOLD_CROSSED', cdg, zone); }
  }

  await chrome.storage.local.set({
    cdgCurrent: cdg, sfiCurrent: sfi, drCurrent: dr, zoneCurrent: zone,
    cdgHistory: session.cdgHistory, sfiHistory: session.sfiHistory,
    drHistory: session.drHistory, zoneHistory: session.zoneHistory,
    ipEvents: session.ipEvents, alertEvents: session.alertEvents,
    tick: session.tick, signalsNorm: norm, signalsRaw: signals,
    sessionStart: session.startTime,
  });

  console.log('[CDG] t=' + session.tick + ' CDG=' + cdg.toFixed(3) + ' zone=' + zone + ' SFI=' + sfi.toFixed(2));
});

function fireAlert(type, cdg, zone) {
  const titles = { INFLECTION_POINT: 'CDG - Inflection Point', THRESHOLD_CROSSED: 'CDG Alert' };
  const msgs   = { INFLECTION_POINT: 'Drift accelerating. CDG=' + cdg.toFixed(3), THRESHOLD_CROSSED: 'CDG crossed 0.60. CDG=' + cdg.toFixed(3) };
  chrome.notifications.create({ type: 'basic', iconUrl: '../assets/icon48.png', title: titles[type], message: msgs[type], priority: 2 });
}

function buildCalibration(windowSignals) {
  const bounds = {};
  for (const k of Object.keys(WEIGHTS)) {
    const vals = windowSignals.map(s => s[k]).filter(v => v !== undefined && !isNaN(v));
    if (vals.length === 0 || Math.max(...vals) - Math.min(...vals) < 0.05) {
      bounds[k] = POPULATION_BOUNDS[k];
    } else {
      bounds[k] = { min: Math.min(...vals), max: Math.max(...vals) };
    }
  }
  return bounds;
}

chrome.runtime.onMessage.addListener((msg, sender, respond) => {
  if (msg.type === 'RESET_SESSION') {
    session = { startTime: Date.now(), tick: 0, cdgHistory: [], sfiHistory: [], drHistory: [], zoneHistory: [], ipEvents: [], alertEvents: [], cdgPrev: 0.0, drPrev: 0.0, calibration: null, calibrationWindow: [], CALIBRATION_TICKS: 10 };
    chrome.storage.local.clear();
    respond({ status: 'reset' });
  }
});

console.log('[CDG] Service worker loaded.');

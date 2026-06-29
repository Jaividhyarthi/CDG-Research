/**
 * CDG content.js — Passive behavioral signal capture
 * Injected into claude.ai / chatgpt.com / gemini.google.com
 * Captures METADATA ONLY — never reads query or response content.
 * Privacy-preserving by design.
 */

(function() {
  'use strict';

  // ── State ──────────────────────────────────────────────────────────────────
  const state = {
    sessionStart:     Date.now(),
    lastQueryTime:    null,
    lastResponseTime: null,
    lastKeyTime:      null,
    keyIntervals:     [],
    errorCount:       0,
    totalKeys:        0,
    queryCount:       0,
    responseCount:    0,
    copyCount:        0,
    modifyCount:      0,
    lastCopyText:     null,
    queryIntervals:   [],
    scrollDepths:     [],
    evalTimes:        [],
    overrideCount:    0,
    baselineKeySpeed: null,
    baselineErrorRate:null,
    calibrated:       false,
    calibrationStart: Date.now(),
    CALIBRATION_MS:   10 * 60 * 1000, // 10 minutes
  };

  // ── Platform detection ─────────────────────────────────────────────────────
  function getPlatform() {
    const h = window.location.hostname;
    if (h.includes('claude.ai'))   return 'claude';
    if (h.includes('chatgpt.com')) return 'chatgpt';
    if (h.includes('gemini'))      return 'gemini';
    return 'unknown';
  }

  // ── Selector map per platform ──────────────────────────────────────────────
  const SELECTORS = {
    claude: {
      input:    'div[contenteditable="true"], textarea',
      submit:   'button[aria-label*="Send"], button[type="submit"]',
      response: '[data-testid="assistant-message"], .font-claude-message',
    },
    chatgpt: {
      input:    'div#prompt-textarea, textarea',
      submit:   'button[data-testid="send-button"]',
      response: '[data-message-author-role="assistant"]',
    },
    gemini: {
      input:    'rich-textarea, textarea',
      submit:   'button.send-button',
      response: '.model-response-text',
    },
  };

  const platform = getPlatform();
  const sel = SELECTORS[platform] || SELECTORS.claude;

  // ── Signal helpers ─────────────────────────────────────────────────────────

  /** PQAR — did user type anything before submitting? */
  function computePQAR() {
    if (state.queryCount === 0) return 0.8;
    // Approximate: if typing intervals exist before queries, user tried first
    const typedBeforeQuery = state.keyIntervals.length > 0 ? 1.0 : 0.3;
    return Math.min(1.0, typedBeforeQuery);
  }

  /** QCS — query complexity (length + structure proxy) */
  function computeQCS(queryLength) {
    if (!queryLength) return 0.5;
    // Longer, more specific queries = higher complexity = more autonomous
    if (queryLength > 200) return 0.9;
    if (queryLength > 100) return 0.7;
    if (queryLength > 50)  return 0.5;
    if (queryLength > 20)  return 0.3;
    return 0.1; // very short = low complexity
  }

  /** TTQ — time from session/task start to first query */
  function computeTTQ() {
    if (!state.lastQueryTime) return 0.5;
    const elapsed = (state.lastQueryTime - state.sessionStart) / 1000; // seconds
    // More time before querying = more autonomous (tried first)
    if (elapsed > 120) return 0.9;
    if (elapsed > 60)  return 0.7;
    if (elapsed > 30)  return 0.5;
    if (elapsed > 10)  return 0.3;
    return 0.1; // immediate query = dependent
  }

  /** ARWM — acceptance rate without modification */
  function computeARWM() {
    if (state.copyCount === 0) return 0.1;
    // ratio of copies that were NOT followed by modification
    const unmodified = Math.max(0, state.copyCount - state.modifyCount);
    return Math.min(1.0, unmodified / state.copyCount);
  }

  /** RET — response evaluation time (normalized) */
  function computeRET() {
    if (state.evalTimes.length === 0) return 0.5;
    const avgEval = state.evalTimes.reduce((a,b)=>a+b,0) / state.evalTimes.length;
    // More time reading = more autonomous
    if (avgEval > 30000) return 0.9; // > 30s
    if (avgEval > 15000) return 0.7;
    if (avgEval > 8000)  return 0.5;
    if (avgEval > 3000)  return 0.3;
    return 0.1; // < 3s = barely read it
  }

  /** OCR — override / correction rate */
  function computeOCR() {
    if (state.queryCount === 0) return 0.3;
    return Math.min(1.0, state.overrideCount / Math.max(1, state.queryCount));
  }

  /** TSD — typing speed decline (0=no decline, 1=severe decline) */
  function computeTSD() {
    if (!state.calibrated || state.keyIntervals.length < 5) return 0.0;
    const recent = state.keyIntervals.slice(-10);
    const avgRecent = recent.reduce((a,b)=>a+b,0) / recent.length;
    if (!state.baselineKeySpeed) return 0.0;
    // Higher interval = slower typing = more fatigue
    const decline = (avgRecent - state.baselineKeySpeed) / state.baselineKeySpeed;
    return Math.min(1.0, Math.max(0.0, decline));
  }

  /** TER — typing error rate (backspace/delete ratio) */
  function computeTER() {
    if (state.totalKeys === 0) return 0.0;
    return Math.min(1.0, state.errorCount / state.totalKeys);
  }

  /** QIC — query interval compression */
  function computeQIC() {
    if (state.queryIntervals.length < 2) return 0.0;
    const recent = state.queryIntervals.slice(-5);
    const avg = recent.reduce((a,b)=>a+b,0) / recent.length;
    // Shorter intervals = rapid fire = more dependent
    if (avg < 10000)  return 0.9; // < 10s between queries
    if (avg < 30000)  return 0.6;
    if (avg < 60000)  return 0.3;
    return 0.0;
  }

  // ── Event listeners ────────────────────────────────────────────────────────

  /** Keystroke timing — captures intervals and error rate */
  document.addEventListener('keydown', (e) => {
    const now = Date.now();
    if (state.lastKeyTime) {
      const interval = now - state.lastKeyTime;
      if (interval > 50 && interval < 5000) { // filter outliers
        state.keyIntervals.push(interval);
        if (state.keyIntervals.length > 100) state.keyIntervals.shift();
      }
    }
    state.lastKeyTime = now;
    state.totalKeys++;

    // Error detection (backspace/delete)
    if (e.key === 'Backspace' || e.key === 'Delete') {
      state.errorCount++;
    }

    // Calibration window
    const elapsed = now - state.calibrationStart;
    if (!state.calibrated && elapsed >= state.CALIBRATION_MS) {
      if (state.keyIntervals.length >= 10) {
        const baseline = state.keyIntervals.slice(0, 20);
        state.baselineKeySpeed = baseline.reduce((a,b)=>a+b,0) / baseline.length;
        state.baselineErrorRate = state.errorCount / Math.max(1, state.totalKeys);
        state.calibrated = true;
        console.log('[CDG] Calibration complete. Baseline key speed:', state.baselineKeySpeed.toFixed(0), 'ms');
      }
    }
  }, true);

  /** Copy event — ARWM detection */
  document.addEventListener('copy', (e) => {
    state.copyCount++;
    state.lastCopyTime = Date.now();
    // We don't read what was copied — only that a copy occurred
    // Track timing to detect if user edits after copying
    setTimeout(() => {
      // If user types within 5 seconds of copying, count as modification
      if (state.lastKeyTime && (Date.now() - state.lastKeyTime) < 5000) {
        state.modifyCount++;
      }
    }, 5000);
  }, true);

  /** Submit detection — TTQ and QCS */
  function attachSubmitListeners() {
    const submitBtns = document.querySelectorAll(sel.submit);
    submitBtns.forEach(btn => {
      if (btn._cdgAttached) return;
      btn._cdgAttached = true;
      btn.addEventListener('click', () => {
        const now = Date.now();
        // TTQ
        if (state.queryCount > 0 && state.lastQueryTime) {
          state.queryIntervals.push(now - state.lastQueryTime);
          if (state.queryIntervals.length > 20) state.queryIntervals.shift();
        }
        state.lastQueryTime = now;
        state.queryCount++;
        // QCS — query length from input
        const input = document.querySelector(sel.input);
        const qLen  = input ? (input.innerText || input.value || '').length : 0;
        // Response eval timer starts when response appears
        state.lastSubmitTime = now;
      });
    });
  }

  /** Response appearance — RET and scroll depth */
  function watchResponses() {
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
          if (node.nodeType !== 1) continue;
          const isResponse = node.matches && (
            node.matches(sel.response) ||
            node.querySelector && node.querySelector(sel.response)
          );
          if (isResponse) {
            const responseTime = Date.now();
            state.responseCount++;
            state.lastResponseTime = responseTime;

            // RET — time until user interacts after response
            const evalStart = responseTime;
            const evalListener = () => {
              const evalTime = Date.now() - evalStart;
              if (evalTime > 500) { // ignore accidental clicks
                state.evalTimes.push(evalTime);
                if (state.evalTimes.length > 20) state.evalTimes.shift();
              }
              document.removeEventListener('keydown', evalListener);
              document.removeEventListener('click',   evalListener);
            };
            setTimeout(() => {
              document.addEventListener('keydown', evalListener, { once: true });
              document.addEventListener('click',   evalListener, { once: true });
            }, 500);

            // Scroll depth
            const responseEl = node.matches && node.matches(sel.response)
              ? node
              : node.querySelector && node.querySelector(sel.response);
            if (responseEl) {
              watchScrollDepth(responseEl);
            }

            // Re-attach submit listeners (DOM may have changed)
            attachSubmitListeners();
          }
        }
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  /** Scroll depth tracking */
  function watchScrollDepth(el) {
    let maxScroll = 0;
    const handler = () => {
      const scrolled  = el.scrollTop || window.scrollY;
      const total     = el.scrollHeight - el.clientHeight;
      const depth     = total > 0 ? scrolled / total : 1.0;
      maxScroll = Math.max(maxScroll, depth);
    };
    window.addEventListener('scroll', handler, { passive: true });
    el.addEventListener('scroll',    handler, { passive: true });
    // Record after 30s
    setTimeout(() => {
      state.scrollDepths.push(maxScroll);
      if (state.scrollDepths.length > 20) state.scrollDepths.shift();
      window.removeEventListener('scroll', handler);
    }, 30000);
  }

  // ── Override detection ─────────────────────────────────────────────────────
  // Detect if user edits/challenges AI response (types after reading)
  let lastResponseInteraction = 0;
  document.addEventListener('keydown', () => {
    if (state.lastResponseTime &&
        Date.now() - state.lastResponseTime < 60000 &&
        Date.now() - state.lastResponseTime > 2000) {
      // User typed within 60s of response = possible correction
      if (Date.now() - lastResponseInteraction > 10000) {
        state.overrideCount++;
        lastResponseInteraction = Date.now();
      }
    }
  }, true);

  // ── Signal aggregation and push ────────────────────────────────────────────
  function getSignals() {
    const input = document.querySelector(sel.input);
    const qLen  = input ? (input.innerText || input.value || '').length : 0;
    return {
      PQAR: computePQAR(),
      QCS:  computeQCS(qLen),
      TTQ:  computeTTQ(),
      ARWM: computeARWM(),
      RET:  computeRET(),
      OCR:  computeOCR(),
      TSD:  computeTSD(),
      TER:  computeTER(),
      QIC:  computeQIC(),
      meta: {
        platform:      platform,
        queryCount:    state.queryCount,
        responseCount: state.responseCount,
        copyCount:     state.copyCount,
        calibrated:    state.calibrated,
        sessionMs:     Date.now() - state.sessionStart,
      }
    };
  }

  // Push signals to background every 60 seconds
  setInterval(() => {
    try {
      if (!chrome.runtime?.id) return; // context invalidated check
      const signals = getSignals();
      chrome.runtime.sendMessage({
        type:    'CDG_SIGNALS',
        signals: signals,
        ts:      Date.now(),
      }).catch(() => {});
    } catch(e) { /* context invalidated — ignore */ }
  }, 60000);

  // Also push on demand
  try {
    chrome.runtime.onMessage.addListener((msg, sender, respond) => {
      if (msg.type === 'GET_SIGNALS') {
        respond(getSignals());
        return true;
      }
    });
  } catch(e) { /* context invalidated — ignore */ }

  // Init
  attachSubmitListeners();
  watchResponses();

  console.log(`[CDG] Content script active on ${platform}. Calibration: ${state.CALIBRATION_MS/60000} min window.`);

})();
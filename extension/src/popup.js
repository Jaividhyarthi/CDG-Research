/**
 * CDG popup.js — Dashboard UI logic
 */

const ZONE_COLORS = {
  low:      '#10b981',
  moderate: '#3b82f6',
  high:     '#f59e0b',
  critical: '#ef4444',
};

const SIGNALS = ['PQAR','QCS','TTQ','ARWM','RET','OCR'];
const WEIGHTS = {PQAR:0.20,QCS:0.15,TTQ:0.10,ARWM:0.25,RET:0.15,OCR:0.15};

function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`[onclick="showTab('${name}')"]`).classList.add('active');
  document.getElementById(`panel-${name}`).classList.add('active');
}

async function loadData() {
  const data = await chrome.storage.local.get(null);
  const hasData = data.cdgHistory && data.cdgHistory.length > 0;

  document.getElementById('no-data').style.display       = hasData ? 'none'  : 'block';
  document.getElementById('tabs').style.display          = hasData ? 'flex'  : 'none';
  document.getElementById('footer').style.display        = hasData ? 'flex'  : 'none';
  document.getElementById('score-section').style.opacity = hasData ? '1'     : '0.4';

  if (!hasData) return;

  const cdg  = data.cdgCurrent  || 0;
  const sfi  = data.sfiCurrent  || 1;
  const dr   = data.drCurrent   || 0;
  const zone = data.zoneCurrent || 'low';
  const tick = data.tick        || 0;
  const zc   = ZONE_COLORS[zone] || ZONE_COLORS.low;

  const raw = data.signalsRaw || {};
  document.getElementById('platform-badge').textContent =
    (raw.meta && raw.meta.platform) ? raw.meta.platform.toUpperCase() : 'CDG';

  const circle = document.getElementById('score-circle');
  circle.style.borderColor = zc;
  document.getElementById('score-num').textContent  = cdg.toFixed(3);
  document.getElementById('score-num').style.color  = zc;
  document.getElementById('score-zone').textContent = zone.toUpperCase();
  document.getElementById('score-zone').style.color = zc;

  const bar = document.getElementById('cdg-bar');
  bar.style.width      = (cdg * 100) + '%';
  bar.style.background = zc;
  document.getElementById('cdg-pct').textContent = (cdg * 100).toFixed(1) + '%';

  document.getElementById('sfi-val').textContent = sfi.toFixed(2);
  document.getElementById('dr-val').textContent  = (dr >= 0 ? '+' : '') + dr.toFixed(3);
  document.getElementById('dr-val').style.color  =
    dr > 0.005 ? '#ef4444' : dr < -0.005 ? '#10b981' : '#3b82f6';
  document.getElementById('tick-val').textContent = tick;

  drawChart(data.cdgHistory || []);
  renderSignals(data.signalsNorm || {}, data.tick || 0);
  renderEvents(data.ipEvents || [], data.alertEvents || []);
  renderStats(data);
}

function drawChart(history) {
  const canvas = document.getElementById('cdg-chart');
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, 0, W, H);
  if (history.length < 2) return;

  const zones = [
    {lo:0.8, hi:1.0, color:'rgba(239,68,68,0.08)'},
    {lo:0.6, hi:0.8, color:'rgba(245,158,11,0.06)'},
    {lo:0.3, hi:0.6, color:'rgba(59,130,246,0.05)'},
    {lo:0.0, hi:0.3, color:'rgba(16,185,129,0.05)'},
  ];
  zones.forEach(({lo, hi, color}) => {
    ctx.fillStyle = color;
    ctx.fillRect(0, H*(1-hi), W, H*(hi-lo));
  });

  ctx.setLineDash([4,4]);
  ctx.strokeStyle = 'rgba(239,68,68,0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, H*0.4);
  ctx.lineTo(W, H*0.4);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.beginPath();
  ctx.strokeStyle = '#8b5cf6';
  ctx.lineWidth = 2;
  history.forEach((v, i) => {
    const x = (i / Math.max(history.length-1, 1)) * W;
    const y = H - (v * H);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.beginPath();
  history.forEach((v, i) => {
    const x = (i / Math.max(history.length-1, 1)) * W;
    const y = H - (v * H);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
  ctx.fillStyle = 'rgba(139,92,246,0.12)';
  ctx.fill();

  const lastVal = history[history.length-1];
  ctx.beginPath();
  ctx.arc(W-4, H-(lastVal*H), 4, 0, Math.PI*2);
  ctx.fillStyle = ZONE_COLORS[getZone(lastVal)];
  ctx.fill();
}

function getZone(v) {
  if (v >= 0.8) return 'critical';
  if (v >= 0.6) return 'high';
  if (v >= 0.3) return 'moderate';
  return 'low';
}

function renderSignals(norm, tick) {
  document.getElementById('cal-banner').style.display = tick < 10 ? 'block' : 'none';
  const list = document.getElementById('signals-list');
  list.innerHTML = '';
  SIGNALS.forEach(k => {
    const val   = norm[k] || 0;
    const color = val > 0.6 ? '#ef4444' : val > 0.3 ? '#f59e0b' : '#10b981';
    const row   = document.createElement('div');
    row.className = 'signal-row';
    row.innerHTML = `
      <span class="signal-name">${k}</span>
      <div class="signal-bar-track">
        <div class="signal-bar-fill" style="width:${val*100}%;background:${color}"></div>
      </div>
      <span class="signal-val" style="color:${color}">${val.toFixed(2)}</span>
      <span class="signal-w">w=${WEIGHTS[k]}</span>
    `;
    list.appendChild(row);
  });
}

function renderEvents(ipEvents, alertEvents) {
  const list = document.getElementById('events-list');
  list.innerHTML = '';
  const all = [
    ...ipEvents.map(t    => ({t, type:'ip',    label:`t=${t} min — Inflection Point detected`})),
    ...alertEvents.map(t => ({t, type:'alert', label:`t=${t} min — Alert: CDG crossed 0.60`})),
  ].sort((a,b) => b.t - a.t);
  if (!all.length) {
    list.innerHTML = '<div class="event-empty">No events yet. Monitoring...</div>';
    return;
  }
  all.forEach(({type, label}) => {
    const div = document.createElement('div');
    div.className = `event-item event-${type}`;
    div.textContent = label;
    list.appendChild(div);
  });
}

function renderStats(data) {
  const h   = data.cdgHistory || [];
  const raw = (data.signalsRaw && data.signalsRaw.meta) || {};
  const stats = [
    ['Session time',    `${data.tick || 0} minutes`],
    ['CDG current',     (data.cdgCurrent || 0).toFixed(4)],
    ['CDG mean',        h.length ? (h.reduce((a,b)=>a+b)/h.length).toFixed(4) : '—'],
    ['CDG max',         h.length ? Math.max(...h).toFixed(4) : '—'],
    ['SFI',             (data.sfiCurrent || 1).toFixed(3)],
    ['Zone',            (data.zoneCurrent || '—').toUpperCase()],
    ['IP events',       (data.ipEvents || []).length],
    ['Alerts',          (data.alertEvents || []).length],
    ['Queries',         raw.queryCount || 0],
    ['Copies detected', raw.copyCount  || 0],
    ['Calibrated',      raw.calibrated ? 'Yes' : 'No (first 10 min)'],
    ['Platform',        raw.platform   || '—'],
  ];
  document.getElementById('stats-list').innerHTML = stats.map(([l,v]) =>
    `<div class="stat-row">
       <span class="stat-label">${l}</span>
       <span class="stat-val">${v}</span>
     </div>`
  ).join('');
}

async function exportSession() {
  const data = await chrome.storage.local.get(null);
  const sid  = `CDG_EXT_${new Date().toISOString().replace(/[:.]/g,'_')}`;
  const blob = new Blob([JSON.stringify({session_id: sid, ...data}, null, 2)],
                        {type: 'application/json'});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = `${sid}.json`; a.click();
  URL.revokeObjectURL(url);
}

async function resetSession() {
  if (!confirm('Reset CDG session? All current data will be cleared.')) return;
  await chrome.runtime.sendMessage({type: 'RESET_SESSION'});
  await chrome.storage.local.clear();
  loadData();
}

loadData();
setInterval(loadData, 5000);
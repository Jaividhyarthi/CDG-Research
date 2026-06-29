const ZONE_COLORS = { low:'#10b981', moderate:'#3b82f6', high:'#f59e0b', critical:'#ef4444' };
const SIGNALS = ['PQAR','QCS','TTQ','ARWM','RET','OCR'];
const WEIGHTS = {PQAR:0.20,QCS:0.15,TTQ:0.10,ARWM:0.25,RET:0.15,OCR:0.15};

function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  document.getElementById('panel-'+name).classList.add('active');
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
  document.getElementById('dr-val').style.color  = dr > 0.005 ? '#ef4444' : dr < -0.005 ? '#10b981' : '#3b82f6';
  document.getElementById('tick-val').textContent = tick;

  drawChart(data.cdgHistory || []);
  renderSignals(data.signalsNorm || {}, tick);
  renderEvents(data.ipEvents || [], data.alertEvents || []);
  renderStats(data);
}

function drawChart(history) {
  const canvas = document.getElementById('cdg-chart');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#111827';
  ctx.fillRect(0,0,W,H);
  if (history.length < 2) return;

  [{lo:0.8,hi:1.0,c:'rgba(239,68,68,0.08)'},{lo:0.6,hi:0.8,c:'rgba(245,158,11,0.06)'},
   {lo:0.3,hi:0.6,c:'rgba(59,130,246,0.05)'},{lo:0.0,hi:0.3,c:'rgba(16,185,129,0.05)'}]
  .forEach(function(z) { ctx.fillStyle=z.c; ctx.fillRect(0,H*(1-z.hi),W,H*(z.hi-z.lo)); });

  ctx.setLineDash([4,4]);
  ctx.strokeStyle = 'rgba(239,68,68,0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0,H*0.4); ctx.lineTo(W,H*0.4); ctx.stroke();
  ctx.setLineDash([]);

  ctx.beginPath();
  ctx.strokeStyle = '#8b5cf6';
  ctx.lineWidth = 2;
  history.forEach(function(v,i) {
    var x=(i/Math.max(history.length-1,1))*W, y=H-(v*H);
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  });
  ctx.stroke();

  ctx.beginPath();
  history.forEach(function(v,i) {
    var x=(i/Math.max(history.length-1,1))*W, y=H-(v*H);
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  });
  ctx.lineTo(W,H); ctx.lineTo(0,H); ctx.closePath();
  ctx.fillStyle = 'rgba(139,92,246,0.12)';
  ctx.fill();

  var lv = history[history.length-1];
  var lz = lv>=0.8 ? 'critical' : lv>=0.6 ? 'high' : lv>=0.3 ? 'moderate' : 'low';
  ctx.beginPath();
  ctx.arc(W-4, H-(lv*H), 4, 0, Math.PI*2);
  ctx.fillStyle = ZONE_COLORS[lz];
  ctx.fill();
}

function renderSignals(norm, tick) {
  var el = document.getElementById('cal-banner');
  if (el) el.style.display = tick < 10 ? 'block' : 'none';
  var list = document.getElementById('signals-list');
  list.innerHTML = '';
  SIGNALS.forEach(function(k) {
    var val = norm[k] || 0;
    var color = val > 0.6 ? '#ef4444' : val > 0.3 ? '#f59e0b' : '#10b981';
    var row = document.createElement('div');
    row.className = 'signal-row';
    row.innerHTML =
      '<span class="signal-name">'+k+'</span>' +
      '<div class="signal-bar-track"><div class="signal-bar-fill" style="width:'+
      (val*100)+'%;background:'+color+'"></div></div>' +
      '<span class="signal-val" style="color:'+color+'">'+val.toFixed(2)+'</span>' +
      '<span class="signal-w">w='+WEIGHTS[k]+'</span>';
    list.appendChild(row);
  });
}

function renderEvents(ipEvents, alertEvents) {
  var list = document.getElementById('events-list');
  list.innerHTML = '';
  var all = [];
  ipEvents.forEach(function(t) { all.push({t:t, type:'ip', label:'t='+t+' min - Inflection Point detected'}); });
  alertEvents.forEach(function(t) { all.push({t:t, type:'alert', label:'t='+t+' min - Alert: CDG crossed 0.60'}); });
  all.sort(function(a,b) { return b.t - a.t; });
  if (!all.length) {
    list.innerHTML = '<div class="event-empty">No events yet. Monitoring...</div>';
    return;
  }
  all.forEach(function(item) {
    var div = document.createElement('div');
    div.className = 'event-item event-'+item.type;
    div.textContent = item.label;
    list.appendChild(div);
  });
}

function renderStats(data) {
  var h = data.cdgHistory || [];
  var raw = (data.signalsRaw && data.signalsRaw.meta) || {};
  var mean = h.length ? (h.reduce(function(a,b){return a+b;},0)/h.length).toFixed(4) : '-';
  var max  = h.length ? Math.max.apply(null,h).toFixed(4) : '-';
  var stats = [
    ['Session time',  (data.tick||0)+' minutes'],
    ['CDG current',   (data.cdgCurrent||0).toFixed(4)],
    ['CDG mean',      mean],
    ['CDG max',       max],
    ['SFI',           (data.sfiCurrent||1).toFixed(3)],
    ['Zone',          (data.zoneCurrent||'-').toUpperCase()],
    ['IP events',     (data.ipEvents||[]).length],
    ['Alerts',        (data.alertEvents||[]).length],
    ['Queries',       raw.queryCount||0],
    ['Copies',        raw.copyCount||0],
    ['Calibrated',    raw.calibrated ? 'Yes' : 'No (first 10 min)'],
    ['Platform',      raw.platform||'-'],
  ];
  document.getElementById('stats-list').innerHTML = stats.map(function(s) {
    return '<div class="stat-row"><span class="stat-label">'+s[0]+'</span><span class="stat-val">'+s[1]+'</span></div>';
  }).join('');
}

async function exportSession() {
  var data = await chrome.storage.local.get(null);
  var sid = 'CDG_EXT_'+new Date().toISOString().replace(/[:.]/g,'_');
  var blob = new Blob([JSON.stringify(Object.assign({session_id:sid},data),null,2)],{type:'application/json'});
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href=url; a.download=sid+'.json'; a.click();
  URL.revokeObjectURL(url);
}

async function resetSession() {
  if (!confirm('Reset CDG session?')) return;
  await chrome.runtime.sendMessage({type:'RESET_SESSION'});
  await chrome.storage.local.clear();
  loadData();
}

// Attach tab listeners
['live','signals','events','stats'].forEach(function(name) {
  var el = document.getElementById('tab-'+name);
  if (el) el.addEventListener('click', function() { showTab(name); });
});

var btnExport = document.getElementById('btn-export');
var btnReset  = document.getElementById('btn-reset');
if (btnExport) btnExport.addEventListener('click', exportSession);
if (btnReset)  btnReset.addEventListener('click', resetSession);

loadData();
setInterval(loadData, 5000);
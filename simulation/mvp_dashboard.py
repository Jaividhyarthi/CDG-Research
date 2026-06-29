"""
CDG MVP Dashboard v2 — simulation/mvp_dashboard.py
Professional research dashboard with menu, history, and full analytics.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons
import sys, os, json, time
from datetime import datetime
import glob

sys.path.insert(0, os.path.dirname(__file__))
from cdg_formula import (
    compute_sfi, compute_cdg, compute_drift_rate,
    detect_inflection_point, get_zone, normalize_signal, ALERT_THRESHOLD
)
from session_generator import (
    autonomous_signals, gradual_drifter_signals,
    threshold_drifter_signals, N_TICKS, SESSION_MINS,
    TICK_INTERVAL, POP_BOUNDS
)

# ── Theme ─────────────────────────────────────────────────────────────────────
BG      = '#0b0f1a'
SURFACE = '#111827'
CARD    = '#1a2235'
BORDER  = '#1e2d45'
TEXT    = '#e2e8f0'
MUTED   = '#64748b'
SOFT    = '#94a3b8'
ACCENT  = '#8b5cf6'
ACCENT2 = '#06b6d4'
GREEN   = '#10b981'
RED     = '#ef4444'
GOLD    = '#f59e0b'
BLUE    = '#3b82f6'

ZONE_COLORS = {'low': GREEN, 'moderate': BLUE, 'high': GOLD, 'critical': RED}
ZONE_RANGES = {'low': (0.0,0.3), 'moderate': (0.3,0.6), 'high': (0.6,0.8), 'critical': (0.8,1.0)}

SIGNAL_LABELS = {
    'PQAR': 'Pre-Query Attempt Rate',
    'QCS':  'Query Complexity Score',
    'TTQ':  'Time to Query',
    'ARWM': 'Acceptance w/o Modification',
    'RET':  'Response Eval Time',
    'OCR':  'Override/Correction Rate',
}
SIGNAL_WEIGHTS = {'PQAR':0.20,'QCS':0.15,'TTQ':0.10,'ARWM':0.25,'RET':0.15,'OCR':0.15}
SIGNALS = ['PQAR','QCS','TTQ','ARWM','RET','OCR']

PILOT_DIR = os.path.join(os.path.dirname(__file__),'..','pilot','sessions')
os.makedirs(PILOT_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': SURFACE,
    'axes.edgecolor': BORDER, 'axes.labelcolor': SOFT,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': TEXT, 'grid.color': BORDER,
    'grid.linewidth': 0.5, 'font.size': 9,
    'font.family': 'DejaVu Sans',
})


# ── Session state ─────────────────────────────────────────────────────────────
class CDGSession:
    def __init__(self, profile):
        self.profile = profile
        self.rng = np.random.default_rng(int(time.time()))
        self.tick = 0
        self.cdg_history = []
        self.sfi_history = []
        self.dr_history  = []
        self.zone_history = []
        self.ip_events   = []
        self.alert_events = []
        self.signal_history = {k: [] for k in POP_BOUNDS}
        self.cdg_prev = 0.0
        self.dr_prev  = 0.0
        self.start_time = datetime.now()
        self.complete = False
        self.signal_fn = {
            'autonomous': autonomous_signals,
            'gradual_drifter': gradual_drifter_signals,
            'threshold_drifter': threshold_drifter_signals,
        }[profile]

    def step(self):
        if self.tick >= N_TICKS:
            self.complete = True
            return None
        t   = self.tick * TICK_INTERVAL
        raw = self.signal_fn(self.tick, N_TICKS, self.rng)
        norm = {k: normalize_signal(k, raw[k], *POP_BOUNDS[k]) for k in POP_BOUNDS}
        sfi  = compute_sfi(t=t, T_max=SESSION_MINS,
                           TSD=raw['TSD'], TER=raw['TER'], QIC=raw['QIC'])
        cdg  = compute_cdg(norm, sfi)
        dr   = compute_drift_rate(cdg, self.cdg_prev, TICK_INTERVAL)

        ip = False
        if self.tick > 5 and cdg >= 0.35:
            recent = self.cdg_history[-5:]
            if recent and (recent[-1] > recent[0] + 0.08):
                avg_dr = np.mean(self.dr_history[-3:]) if len(self.dr_history) >= 3 else 0
                if avg_dr > 0.008:
                    ip = detect_inflection_point(dr, self.dr_prev, TICK_INTERVAL, cdg_current=cdg)

        zone = get_zone(cdg)
        self.cdg_history.append(cdg)
        self.sfi_history.append(sfi)
        self.dr_history.append(dr)
        self.zone_history.append(zone)
        for k in POP_BOUNDS:
            self.signal_history[k].append(norm[k])
        if ip:
            self.ip_events.append(self.tick)
        if cdg >= ALERT_THRESHOLD and (not self.alert_events or self.tick - self.alert_events[-1] > 10):
            self.alert_events.append(self.tick)
        self.cdg_prev = cdg
        self.dr_prev  = dr
        self.tick += 1
        return {'tick': self.tick, 't_min': t, 'cdg': cdg, 'sfi': sfi,
                'dr': dr, 'zone': zone, 'ip': ip, 'norm': norm}

    def export(self):
        sid = f"MVP_{self.profile}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        data = {
            'session_id': sid, 'profile': self.profile,
            'start_time': self.start_time.isoformat(),
            'n_ticks': self.tick,
            'cdg_mean': float(np.mean(self.cdg_history)) if self.cdg_history else 0,
            'cdg_max':  float(np.max(self.cdg_history))  if self.cdg_history else 0,
            'cdg_final': float(self.cdg_history[-1])     if self.cdg_history else 0,
            'sfi_mean': float(np.mean(self.sfi_history)) if self.sfi_history else 0,
            'ip_events': self.ip_events,
            'alert_events': self.alert_events,
            'final_zone': self.zone_history[-1] if self.zone_history else 'low',
            'cdg_timeseries': [round(v,4) for v in self.cdg_history],
            'sfi_timeseries': [round(v,4) for v in self.sfi_history],
        }
        path = os.path.join(PILOT_DIR, f"{sid}.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return data, path


# ── Load history ──────────────────────────────────────────────────────────────
def load_history():
    files = sorted(glob.glob(os.path.join(PILOT_DIR, '*.json')), reverse=True)
    sessions = []
    for f in files[:10]:
        try:
            with open(f) as fp:
                sessions.append(json.load(fp))
        except:
            pass
    return sessions


# ══════════════════════════════════════════════════════════════════════════════
# MENU SCREEN
# ══════════════════════════════════════════════════════════════════════════════
class MenuScreen:
    def __init__(self):
        self.selected_profile = 'gradual_drifter'
        self.result = None
        self._build()

    def _build(self):
        self.fig = plt.figure(figsize=(13, 8), facecolor=BG)
        self.fig.canvas.manager.set_window_title('CDG Research Dashboard')

        # Title
        self.fig.text(0.5, 0.93, 'Cognitive Dependency Gradient',
                      ha='center', fontsize=20, fontweight='bold', color=TEXT)
        self.fig.text(0.5, 0.88, 'CDG(t) = SFI(t) × [w₁·PQAR\' + w₂·QCS\' + w₃·TTQ\' + w₄·ARWM + w₅·RET\' + w₆·OCR\']',
                      ha='center', fontsize=10, color=ACCENT, family='monospace')
        self.fig.text(0.5, 0.84, 'AUC = 1.0000  ·  F1 = 0.9951  ·  IP Precision = 0.9802  ·  False Positive Rate = 0.0%  ·  Discrimination Ratio = 75.9×',
                      ha='center', fontsize=9, color=MUTED)

        # Divider
        line = plt.Line2D([0.05, 0.95], [0.82, 0.82], transform=self.fig.transFigure,
                          color=BORDER, linewidth=1)
        self.fig.add_artist(line)

        gs = gridspec.GridSpec(2, 3, figure=self.fig,
                               left=0.06, right=0.94, top=0.78, bottom=0.08,
                               hspace=0.35, wspace=0.3)

        # ── Profile selector ──
        ax_prof = self.fig.add_subplot(gs[0, 0])
        ax_prof.set_facecolor(CARD)
        for spine in ax_prof.spines.values():
            spine.set_edgecolor(BORDER)
        ax_prof.set_xticks([]); ax_prof.set_yticks([])
        ax_prof.set_title('Select Profile', color=TEXT, pad=8, fontsize=10)

        profiles = [
            ('autonomous',        '1. Autonomous',        GREEN,  'CDG stays low. No alerts.'),
            ('gradual_drifter',   '2. Gradual Drifter',   ACCENT, 'Steady drift. Alert ~min 24.'),
            ('threshold_drifter', '3. Threshold Drifter', GOLD,   'Flat then spikes. Alert ~min 39.'),
        ]
        self._profile_texts = {}
        for i, (key, label, color, desc) in enumerate(profiles):
            y = 0.72 - i * 0.28
            rect = mpatches.FancyBboxPatch(
                (0.05, y - 0.08), 0.90, 0.22,
                boxstyle='round,pad=0.01',
                facecolor=SURFACE, edgecolor=BORDER,
                transform=ax_prof.transAxes, clip_on=False
            )
            ax_prof.add_patch(rect)
            t1 = ax_prof.text(0.12, y + 0.06, label, transform=ax_prof.transAxes,
                              fontsize=9, fontweight='bold', color=color, va='center')
            ax_prof.text(0.12, y - 0.02, desc, transform=ax_prof.transAxes,
                         fontsize=8, color=MUTED, va='center')
            self._profile_texts[key] = (rect, t1, color)

        self._highlight_profile()
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # ── Stats summary ──
        ax_stats = self.fig.add_subplot(gs[0, 1])
        ax_stats.set_facecolor(CARD)
        for spine in ax_stats.spines.values():
            spine.set_edgecolor(BORDER)
        ax_stats.set_xticks([]); ax_stats.set_yticks([])
        ax_stats.set_title('Formula Validation Summary', color=TEXT, pad=8, fontsize=10)

        stats = [
            ('AUC-ROC',             '1.0000', GREEN),
            ('F1 Score',            '0.9951', GREEN),
            ('Precision',           '0.9902', GREEN),
            ('Recall',              '1.0000', GREEN),
            ('RMSE',                '0.0686', GREEN),
            ('IP Precision',        '0.9802', GREEN),
            ('False Positive Rate', '0.0%',   GREEN),
            ('Discrimination',      '75.9×',  GREEN),
            ('Early Warning',       '12.8 min', ACCENT2),
            ('Stability (5 seeds)', 'std=0.000', ACCENT2),
        ]
        for i, (label, val, color) in enumerate(stats):
            y = 0.92 - i * 0.09
            ax_stats.text(0.05, y, label, transform=ax_stats.transAxes,
                          fontsize=8, color=SOFT)
            ax_stats.text(0.72, y, val, transform=ax_stats.transAxes,
                          fontsize=8, color=color, fontweight='bold', ha='right')

        # ── History panel ──
        ax_hist = self.fig.add_subplot(gs[0, 2])
        ax_hist.set_facecolor(CARD)
        for spine in ax_hist.spines.values():
            spine.set_edgecolor(BORDER)
        ax_hist.set_xticks([]); ax_hist.set_yticks([])
        ax_hist.set_title('Recent Sessions', color=TEXT, pad=8, fontsize=10)

        history = load_history()
        if history:
            for i, s in enumerate(history[:7]):
                y = 0.90 - i * 0.13
                profile_short = s['profile'].replace('_drifter','').replace('autonomous','auto')
                zone_color = ZONE_COLORS.get(s.get('final_zone','low'), GREEN)
                dt = s.get('start_time','')[:16].replace('T',' ')
                ax_hist.text(0.04, y, dt, transform=ax_hist.transAxes,
                             fontsize=7, color=MUTED)
                ax_hist.text(0.04, y-0.05, f"{profile_short}  CDG={s['cdg_mean']:.3f}  {s.get('final_zone','?').upper()}",
                             transform=ax_hist.transAxes,
                             fontsize=8, color=zone_color, fontweight='bold')
        else:
            ax_hist.text(0.5, 0.5, 'No sessions yet', transform=ax_hist.transAxes,
                         ha='center', va='center', color=MUTED, fontsize=9)

        # ── History comparison chart ──
        ax_comp = self.fig.add_subplot(gs[1, :2])
        ax_comp.set_facecolor(SURFACE)
        ax_comp.set_title('Session History — CDG Curves Overlay', color=TEXT, pad=8, fontsize=10)
        if history:
            for s in history[:5]:
                curve = s.get('cdg_timeseries', [])
                if curve:
                    color = ZONE_COLORS.get(s.get('final_zone','low'), GREEN)
                    label = f"{s['profile'][:3].upper()} {s['start_time'][11:16]}"
                    ax_comp.plot(curve, color=color, alpha=0.6,
                                 linewidth=1.5, label=label)
            ax_comp.axhline(0.60, color=RED, linestyle='--', linewidth=1, alpha=0.5)
            ax_comp.set_xlabel('Session tick', color=MUTED)
            ax_comp.set_ylabel('CDG Score', color=MUTED)
            ax_comp.set_ylim(0, 1.05)
            ax_comp.legend(fontsize=7, framealpha=0.2, loc='upper left')
            ax_comp.grid(True, alpha=0.3)
        else:
            ax_comp.text(0.5, 0.5, 'Run sessions to see history',
                         transform=ax_comp.transAxes, ha='center', va='center',
                         color=MUTED, fontsize=10)

        # ── Session stats bar ──
        ax_bar = self.fig.add_subplot(gs[1, 2])
        ax_bar.set_facecolor(SURFACE)
        ax_bar.set_title('Sessions by Profile', color=TEXT, pad=8, fontsize=10)
        if history:
            counts = {}
            for s in history:
                p = s['profile']
                counts[p] = counts.get(p, 0) + 1
            colors_bar = [PROFILE_COLOR(p) for p in counts]
            ax_bar.bar([p.replace('_','\n') for p in counts],
                       list(counts.values()),
                       color=colors_bar, alpha=0.8, edgecolor='none')
            ax_bar.set_ylabel('Count', color=MUTED)
            ax_bar.grid(True, alpha=0.3, axis='y')
        else:
            ax_bar.text(0.5, 0.5, 'No data yet',
                        transform=ax_bar.transAxes, ha='center',
                        va='center', color=MUTED)

        # ── Start button ──
        ax_btn = self.fig.add_axes([0.38, 0.01, 0.24, 0.055])
        self.btn_start = Button(ax_btn, 'START SESSION',
                                color=ACCENT, hovercolor='#7c3aed')
        self.btn_start.label.set_color('white')
        self.btn_start.label.set_fontsize(11)
        self.btn_start.label.set_fontweight('bold')
        self.btn_start.on_clicked(self._start)

        self.fig.text(0.5, 0.005,
                      'Click a profile to select  ·  Press START SESSION to begin',
                      ha='center', fontsize=8, color=MUTED)

    def _profile_ax_coords(self, ax, event):
        try:
            inv = ax.transAxes.inverted()
            x, y = inv.transform((event.x, event.y))
            return 0 <= x <= 1 and 0 <= y <= 1
        except:
            return False

    def _on_click(self, event):
        if event.inaxes is None:
            return
        # Detect which profile row was clicked based on y position in axes
        try:
            inv = event.inaxes.transAxes.inverted()
            _, ay = inv.transform((event.x, event.y))
        except:
            return
        profiles_order = ['autonomous', 'gradual_drifter', 'threshold_drifter']
        y_centers = [0.72, 0.44, 0.16]
        for key, yc in zip(profiles_order, y_centers):
            if abs(ay - yc) < 0.14:
                self.selected_profile = key
                self._highlight_profile()
                self.fig.canvas.draw_idle()
                break

    def _highlight_profile(self):
        for key, (rect, t1, color) in self._profile_texts.items():
            if key == self.selected_profile:
                rect.set_edgecolor(color)
                rect.set_linewidth(2)
                rect.set_facecolor(CARD)
            else:
                rect.set_edgecolor(BORDER)
                rect.set_linewidth(1)
                rect.set_facecolor(SURFACE)

    def _start(self, event):
        self.result = self.selected_profile
        plt.close(self.fig)

    def show(self):
        plt.show(block=True)
        return self.result


def PROFILE_COLOR(p):
    return {'autonomous': GREEN, 'gradual_drifter': ACCENT,
            'threshold_drifter': GOLD}.get(p, MUTED)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
class LiveDashboard:
    def __init__(self, profile):
        self.session = CDGSession(profile)
        self.paused  = False
        self.ani     = None
        self._build()

    def _build(self):
        self.fig = plt.figure(figsize=(16, 9), facecolor=BG)
        self.fig.canvas.manager.set_window_title('CDG Live Dashboard')

        gs = gridspec.GridSpec(
            4, 6, figure=self.fig,
            left=0.05, right=0.97,
            top=0.91, bottom=0.09,
            hspace=0.55, wspace=0.40
        )

        # Row 0: CDG main curve (full width)
        self.ax_cdg = self.fig.add_subplot(gs[0, :])

        # Row 1: SFI curve | DR curve | Zone timeline | CDG score card
        self.ax_sfi   = self.fig.add_subplot(gs[1, :2])
        self.ax_dr    = self.fig.add_subplot(gs[1, 2:4])
        self.ax_zone  = self.fig.add_subplot(gs[1, 4:])

        # Row 2: Signal bars | Signal trends
        self.ax_sigbar   = self.fig.add_subplot(gs[2, :2])
        self.ax_sigtrend = self.fig.add_subplot(gs[2, 2:5])
        self.ax_scorecard = self.fig.add_subplot(gs[2, 5])

        # Row 3: Event log | Stats | Weight contribution
        self.ax_log    = self.fig.add_subplot(gs[3, :2])
        self.ax_stats  = self.fig.add_subplot(gs[3, 2:4])
        self.ax_weight = self.fig.add_subplot(gs[3, 4:])

        for ax in [self.ax_log, self.ax_stats, self.ax_scorecard]:
            ax.set_facecolor(CARD)

        self._setup_static()

    def _setup_static(self):
        p = self.session.profile.replace('_',' ').title()
        self.fig.suptitle(
            f'CDG Live Research Dashboard  ·  {p}  ·  '
            f'AUC=1.00  F1=0.995  Discrimination=75.9×',
            color=TEXT, fontsize=11, fontweight='bold', y=0.97
        )

        # CDG axis static setup
        self.ax_cdg.set_xlim(0, N_TICKS)
        self.ax_cdg.set_ylim(-0.02, 1.08)
        self.ax_cdg.set_xlabel('Session time (minutes)', color=MUTED, fontsize=8)
        self.ax_cdg.set_ylabel('CDG Score', color=MUTED, fontsize=8)
        self.ax_cdg.set_title('CDG(t) — Cognitive Dependency Gradient (Live)',
                               color=TEXT, pad=6, fontsize=10)
        for (lo, hi), color in [(ZONE_RANGES[z], ZONE_COLORS[z])
                                 for z in ['low','moderate','high','critical']]:
            self.ax_cdg.axhspan(lo, hi, alpha=0.05, color=color)
        self.ax_cdg.axhline(ALERT_THRESHOLD, color=RED,
                             linestyle='--', linewidth=1, alpha=0.6)
        for y, label in [(0.15,'LOW'),(0.45,'MODERATE'),(0.70,'HIGH'),(0.90,'CRITICAL')]:
            self.ax_cdg.text(N_TICKS-0.5, y, label, color=MUTED,
                              fontsize=7, ha='right', va='center', alpha=0.5)
        self.ax_cdg.grid(True, alpha=0.25)

        # Pause button
        ax_pause = self.fig.add_axes([0.44, 0.01, 0.12, 0.04])
        self.btn_pause = Button(ax_pause, 'PAUSE',
                                color=CARD, hovercolor=BORDER)
        self.btn_pause.label.set_color(SOFT)
        self.btn_pause.label.set_fontsize(9)
        self.btn_pause.on_clicked(self._toggle_pause)

        # Menu button
        ax_menu = self.fig.add_axes([0.57, 0.01, 0.12, 0.04])
        self.btn_menu = Button(ax_menu, 'BACK TO MENU',
                               color=CARD, hovercolor=BORDER)
        self.btn_menu.label.set_color(SOFT)
        self.btn_menu.label.set_fontsize(9)
        self.btn_menu.on_clicked(self._back_menu)

    def _toggle_pause(self, event):
        self.paused = not self.paused
        self.btn_pause.label.set_text('RESUME' if self.paused else 'PAUSE')
        self.fig.canvas.draw_idle()

    def _back_menu(self, event):
        if self.ani:
            self.ani.event_source.stop()
        plt.close(self.fig)
        main()

    def update(self, frame):
        if self.paused or self.session.complete:
            return

        state = self.session.step()
        if state is None:
            self._on_complete()
            return

        cdg  = state['cdg']
        sfi  = state['sfi']
        dr   = state['dr']
        zone = state['zone']
        tick = state['tick']
        norm = state['norm']
        zc   = ZONE_COLORS[zone]

        hist = self.session.cdg_history
        ticks = list(range(len(hist)))

        # ── CDG main curve ────────────────────────────────────────────────────
        self.ax_cdg.cla()
        self._setup_static()
        self.ax_cdg.plot(ticks, hist, color=ACCENT, linewidth=2, alpha=0.9, zorder=3)
        self.ax_cdg.fill_between(ticks, hist, alpha=0.12, color=ACCENT)
        for ip_t in self.session.ip_events:
            if ip_t < len(hist):
                self.ax_cdg.axvline(ip_t, color=GOLD, linestyle=':', linewidth=1.5, alpha=0.8)
                self.ax_cdg.text(ip_t+0.3, hist[ip_t]+0.03, 'IP', fontsize=7, color=GOLD)
        for al_t in self.session.alert_events:
            if al_t < len(hist):
                self.ax_cdg.axvline(al_t, color=RED, linestyle='--', linewidth=1.2, alpha=0.5)
        if hist:
            self.ax_cdg.scatter([tick-1], [cdg], color=zc, s=70, zorder=5)
        self.ax_cdg.text(
            0.01, 0.90,
            f'CDG = {cdg:.3f}   [{zone.upper()}]   t = {tick} / {N_TICKS} min',
            transform=self.ax_cdg.transAxes, fontsize=11, fontweight='bold',
            color=zc,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=CARD, alpha=0.9, edgecolor=zc)
        )

        # ── SFI curve ─────────────────────────────────────────────────────────
        self.ax_sfi.cla()
        self.ax_sfi.set_facecolor(SURFACE)
        if self.session.sfi_history:
            sfi_c = [GOLD if v > 1.7 else BLUE if v > 1.4 else GREEN
                     for v in self.session.sfi_history]
            for i in range(len(self.session.sfi_history)-1):
                self.ax_sfi.plot([i, i+1],
                                 [self.session.sfi_history[i], self.session.sfi_history[i+1]],
                                 color=sfi_c[i], linewidth=1.8)
        self.ax_sfi.axhline(1.0, color=MUTED, linestyle=':', linewidth=0.8)
        self.ax_sfi.axhline(2.0, color=RED, linestyle='--', linewidth=0.8, alpha=0.5)
        self.ax_sfi.set_title(f'Session Fatigue Index  SFI={sfi:.3f}',
                               color=TEXT, fontsize=8, pad=4)
        self.ax_sfi.set_ylim(0.9, 2.1)
        self.ax_sfi.set_xlim(0, N_TICKS)
        self.ax_sfi.set_xlabel('minute', color=MUTED, fontsize=7)
        self.ax_sfi.grid(True, alpha=0.2)

        # ── DR curve ──────────────────────────────────────────────────────────
        self.ax_dr.cla()
        self.ax_dr.set_facecolor(SURFACE)
        if len(self.session.dr_history) > 1:
            dr_arr = np.array(self.session.dr_history)
            dr_x   = list(range(len(dr_arr)))
            self.ax_dr.fill_between(dr_x, dr_arr, 0,
                                     where=dr_arr > 0, color=RED, alpha=0.3)
            self.ax_dr.fill_between(dr_x, dr_arr, 0,
                                     where=dr_arr < 0, color=GREEN, alpha=0.3)
            self.ax_dr.plot(dr_x, dr_arr, color=ACCENT2, linewidth=1.5, alpha=0.8)
        self.ax_dr.axhline(0, color=MUTED, linewidth=0.8)
        dr_label = ('DRIFTING' if dr > 0.005 else
                    'RECOVERING' if dr < -0.005 else 'STABLE')
        dr_color = RED if dr > 0.005 else GREEN if dr < -0.005 else BLUE
        self.ax_dr.set_title(f'Drift Rate  DR={dr:+.4f}  [{dr_label}]',
                              color=dr_color, fontsize=8, pad=4)
        self.ax_dr.set_xlim(0, N_TICKS)
        self.ax_dr.set_xlabel('minute', color=MUTED, fontsize=7)
        self.ax_dr.grid(True, alpha=0.2)

        # ── Zone timeline ─────────────────────────────────────────────────────
        self.ax_zone.cla()
        self.ax_zone.set_facecolor(SURFACE)
        if self.session.zone_history:
            zone_num = {'low':0,'moderate':1,'high':2,'critical':3}
            zn = [zone_num[z] for z in self.session.zone_history]
            zt = list(range(len(zn)))
            self.ax_zone.scatter(zt, zn, c=[ZONE_COLORS[z] for z in self.session.zone_history],
                                  s=8, alpha=0.8, edgecolors='none')
            self.ax_zone.set_yticks([0,1,2,3])
            self.ax_zone.set_yticklabels(['LOW','MOD','HIGH','CRIT'], fontsize=7)
        self.ax_zone.set_title('Zone Timeline', color=TEXT, fontsize=8, pad=4)
        self.ax_zone.set_xlim(0, N_TICKS)
        self.ax_zone.set_xlabel('minute', color=MUTED, fontsize=7)
        self.ax_zone.grid(True, alpha=0.2)

        # ── Signal bars ───────────────────────────────────────────────────────
        self.ax_sigbar.cla()
        self.ax_sigbar.set_facecolor(SURFACE)
        sig_vals = [norm[k] for k in SIGNALS]
        colors_b = [RED if v>0.6 else GOLD if v>0.3 else GREEN for v in sig_vals]
        bars = self.ax_sigbar.barh(
            [f"{k} (w={SIGNAL_WEIGHTS[k]})" for k in SIGNALS],
            sig_vals, color=colors_b, alpha=0.8, edgecolor='none'
        )
        for bar, val in zip(bars, sig_vals):
            self.ax_sigbar.text(
                min(val+0.02, 0.97), bar.get_y()+bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=7, color=TEXT
            )
        self.ax_sigbar.set_xlim(0, 1.05)
        self.ax_sigbar.axvline(0.5, color=MUTED, linestyle='--', linewidth=0.7, alpha=0.5)
        self.ax_sigbar.set_title('Normalized Signals (current tick)',
                                  color=TEXT, fontsize=8, pad=4)
        self.ax_sigbar.tick_params(labelsize=7)
        self.ax_sigbar.grid(True, alpha=0.2, axis='x')

        # ── Signal trends ─────────────────────────────────────────────────────
        self.ax_sigtrend.cla()
        self.ax_sigtrend.set_facecolor(SURFACE)
        sig_colors = [ACCENT, ACCENT2, GREEN, RED, GOLD, BLUE]
        for i, k in enumerate(SIGNALS):
            if len(self.session.signal_history[k]) > 1:
                self.ax_sigtrend.plot(
                    self.session.signal_history[k],
                    color=sig_colors[i], linewidth=1.2, alpha=0.8,
                    label=k
                )
        self.ax_sigtrend.axhline(0.5, color=MUTED, linestyle='--',
                                  linewidth=0.7, alpha=0.4)
        self.ax_sigtrend.set_title('Signal Trends Over Session',
                                    color=TEXT, fontsize=8, pad=4)
        self.ax_sigtrend.set_xlim(0, N_TICKS)
        self.ax_sigtrend.set_ylim(-0.02, 1.05)
        self.ax_sigtrend.legend(fontsize=6, framealpha=0.2,
                                 loc='upper left', ncol=3)
        self.ax_sigtrend.set_xlabel('minute', color=MUTED, fontsize=7)
        self.ax_sigtrend.grid(True, alpha=0.2)

        # ── Score card ────────────────────────────────────────────────────────
        self.ax_scorecard.cla()
        self.ax_scorecard.set_facecolor(CARD)
        self.ax_scorecard.set_xticks([]); self.ax_scorecard.set_yticks([])
        for spine in self.ax_scorecard.spines.values():
            spine.set_edgecolor(zc)
            spine.set_linewidth(1.5)
        self.ax_scorecard.text(0.5, 0.75, f'{cdg:.3f}',
                                transform=self.ax_scorecard.transAxes,
                                ha='center', va='center',
                                fontsize=22, fontweight='bold', color=zc)
        self.ax_scorecard.text(0.5, 0.45, zone.upper(),
                                transform=self.ax_scorecard.transAxes,
                                ha='center', va='center',
                                fontsize=10, fontweight='bold', color=zc)
        self.ax_scorecard.text(0.5, 0.20, f'SFI {sfi:.2f}',
                                transform=self.ax_scorecard.transAxes,
                                ha='center', fontsize=8, color=SOFT)
        self.ax_scorecard.set_title('CDG Now', color=TEXT, fontsize=8, pad=4)

        # ── Event log ─────────────────────────────────────────────────────────
        self.ax_log.cla()
        self.ax_log.set_facecolor(CARD)
        self.ax_log.set_xticks([]); self.ax_log.set_yticks([])
        self.ax_log.set_title('Event Log', color=TEXT, fontsize=8, pad=4)
        log_lines = []
        for ip_t in self.session.ip_events:
            log_lines.append((f't={ip_t:02d}  INFLECTION POINT detected', GOLD))
        for al_t in self.session.alert_events:
            log_lines.append((f't={al_t:02d}  ALERT — CDG crossed 0.60', RED))
        log_lines = sorted(log_lines, key=lambda x: int(x[0][2:4]), reverse=True)
        if not log_lines:
            log_lines = [('Monitoring... no events yet', MUTED)]
        for i, (line, color) in enumerate(log_lines[:6]):
            self.ax_log.text(0.03, 0.88 - i*0.15, line,
                              transform=self.ax_log.transAxes,
                              fontsize=8, color=color, family='monospace')

        # ── Stats ─────────────────────────────────────────────────────────────
        self.ax_stats.cla()
        self.ax_stats.set_facecolor(CARD)
        self.ax_stats.set_xticks([]); self.ax_stats.set_yticks([])
        self.ax_stats.set_title('Session Statistics', color=TEXT, fontsize=8, pad=4)
        stats = [
            ('Progress',    f'{tick}/{N_TICKS} min  ({tick/N_TICKS*100:.0f}%)'),
            ('Mean CDG',    f'{np.mean(hist):.4f}' if hist else '—'),
            ('Max CDG',     f'{np.max(hist):.4f}'  if hist else '—'),
            ('Current CDG', f'{cdg:.4f}'),
            ('SFI',         f'{sfi:.3f}'),
            ('Drift Rate',  f'{dr:+.5f}'),
            ('IP events',   str(len(self.session.ip_events))),
            ('Alerts',      str(len(self.session.alert_events))),
        ]
        for i, (label, value) in enumerate(stats):
            y = 0.90 - i*0.11
            self.ax_stats.text(0.04, y, f'{label}:', transform=self.ax_stats.transAxes,
                                fontsize=8, color=MUTED)
            self.ax_stats.text(0.55, y, value, transform=self.ax_stats.transAxes,
                                fontsize=8, color=TEXT, fontweight='bold')

        # ── Weight contribution ───────────────────────────────────────────────
        self.ax_weight.cla()
        self.ax_weight.set_facecolor(SURFACE)
        contributions = [SIGNAL_WEIGHTS[k] * norm[k] for k in SIGNALS]
        total = sum(contributions) if sum(contributions) > 0 else 1
        pcts  = [c/total*100 for c in contributions]
        wedge_colors = [RED if norm[k]>0.6 else GOLD if norm[k]>0.3 else GREEN
                        for k in SIGNALS]
        if sum(contributions) > 0:
            wedges, texts, autotexts = self.ax_weight.pie(
                contributions, labels=SIGNALS,
                colors=wedge_colors, autopct='%1.0f%%',
                textprops={'color': TEXT, 'fontsize': 7},
                pctdistance=0.75, startangle=90
            )
            for at in autotexts:
                at.set_fontsize(6)
                at.set_color(BG)
        self.ax_weight.set_title('Signal Weight Contribution',
                                  color=TEXT, fontsize=8, pad=4)

    def _on_complete(self):
        if self.ani:
            self.ani.event_source.stop()
        data, path = self.session.export()
        print(f"\n{'='*50}")
        print(f"  SESSION COMPLETE")
        print(f"  CDG final : {data['cdg_final']:.4f}  [{data['final_zone'].upper()}]")
        print(f"  CDG mean  : {data['cdg_mean']:.4f}")
        print(f"  IP events : {len(data['ip_events'])}")
        print(f"  Alerts    : {len(data['alert_events'])}")
        print(f"  Exported  : {path}")
        print(f"{'='*50}\n")

        self.ax_cdg.set_title(
            f"SESSION COMPLETE  —  Final CDG: {data['cdg_final']:.3f}  "
            f"[{data['final_zone'].upper()}]  "
            f"IP events: {len(data['ip_events'])}  Alerts: {len(data['alert_events'])}",
            color=ZONE_COLORS[data['final_zone']], fontsize=10, pad=6
        )
        plt.draw()

    def run(self, speed_ms=250):
        self.ani = FuncAnimation(self.fig, self.update,
                                  interval=speed_ms, cache_frame_data=False)
        plt.show(block=True)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    menu = MenuScreen()
    profile = menu.show()
    if profile is None:
        return
    dashboard = LiveDashboard(profile=profile)
    dashboard.run()


if __name__ == "__main__":
    main()
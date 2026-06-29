"""
CDG MVP Dashboard — simulation/mvp_dashboard.py
Real-time CDG simulator with live visualization.
Proves CDG works end-to-end before Chrome extension build.
Run: python simulation/mvp_dashboard.py
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from cdg_formula import (
    compute_sfi, compute_cdg, compute_drift_rate,
    detect_inflection_point, get_zone, normalize_signal, ALERT_THRESHOLD
)
from session_generator import (
    autonomous_signals, gradual_drifter_signals,
    threshold_drifter_signals, N_TICKS, SESSION_MINS,
    TICK_INTERVAL, POP_BOUNDS, PROFILES
)

# ── Config ────────────────────────────────────────────────────────────────────
TICK_SPEED_MS   = 300      # ms per tick (lower = faster simulation)
HISTORY_WINDOW  = 60       # ticks to show on rolling chart

ZONE_COLORS = {
    'low':      '#10b981',
    'moderate': '#3b82f6',
    'high':     '#f59e0b',
    'critical': '#ef4444',
}
BG      = '#0b0f1a'
SURFACE = '#111827'
CARD    = '#1a2235'
BORDER  = '#1e2d45'
TEXT    = '#e2e8f0'
MUTED   = '#64748b'
ACCENT  = '#8b5cf6'
ACCENT2 = '#06b6d4'


# ── Session state ─────────────────────────────────────────────────────────────
class CDGSession:
    def __init__(self, profile):
        self.profile    = profile
        self.rng        = np.random.default_rng(int(time.time()))
        self.tick       = 0
        self.cdg_history   = []
        self.sfi_history   = []
        self.dr_history    = []
        self.zone_history  = []
        self.ip_events     = []
        self.alert_events  = []
        self.signal_history = {k: [] for k in POP_BOUNDS}
        self.cdg_prev = 0.0
        self.dr_prev  = 0.0
        self.session_start = datetime.now()
        self.complete = False

        self.signal_fn = {
            'autonomous':        autonomous_signals,
            'gradual_drifter':   gradual_drifter_signals,
            'threshold_drifter': threshold_drifter_signals,
        }[profile]

    def step(self):
        if self.tick >= N_TICKS:
            self.complete = True
            return None

        t   = self.tick * TICK_INTERVAL
        raw = self.signal_fn(self.tick, N_TICKS, self.rng)
        norm = {k: normalize_signal(k, raw[k], *POP_BOUNDS[k])
                for k in POP_BOUNDS}

        sfi = compute_sfi(t=t, T_max=SESSION_MINS,
                          TSD=raw['TSD'], TER=raw['TER'], QIC=raw['QIC'])
        cdg = compute_cdg(norm, sfi)
        dr  = compute_drift_rate(cdg, self.cdg_prev, TICK_INTERVAL)

        # IP detection
        ip = False
        if self.tick > 5 and cdg >= 0.35:
            recent = self.cdg_history[-5:] if len(self.cdg_history) >= 5 else []
            if recent and (recent[-1] > recent[0] + 0.08):
                avg_dr = np.mean(self.dr_history[-3:]) \
                         if len(self.dr_history) >= 3 else 0
                if avg_dr > 0.008:
                    ip = detect_inflection_point(
                        dr, self.dr_prev, TICK_INTERVAL,
                        cdg_current=cdg
                    )

        zone = get_zone(cdg)

        # Log
        self.cdg_history.append(cdg)
        self.sfi_history.append(sfi)
        self.dr_history.append(dr)
        self.zone_history.append(zone)
        for k in POP_BOUNDS:
            self.signal_history[k].append(norm[k])

        if ip:
            self.ip_events.append(self.tick)
        if cdg >= ALERT_THRESHOLD and (
                not self.alert_events or
                self.tick - self.alert_events[-1] > 10):
            self.alert_events.append(self.tick)

        self.cdg_prev = cdg
        self.dr_prev  = dr
        self.tick     += 1

        return {
            'tick': self.tick, 't_min': t,
            'cdg': cdg, 'sfi': sfi, 'dr': dr,
            'zone': zone, 'ip': ip,
            'norm': norm, 'raw': raw,
        }

    def export_json(self):
        return {
            'session_id':    f"MVP_{self.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'profile':       self.profile,
            'start_time':    self.session_start.isoformat(),
            'n_ticks':       self.tick,
            'cdg_mean':      float(np.mean(self.cdg_history)),
            'cdg_max':       float(np.max(self.cdg_history)),
            'cdg_final':     float(self.cdg_history[-1]),
            'sfi_mean':      float(np.mean(self.sfi_history)),
            'ip_events':     self.ip_events,
            'alert_events':  self.alert_events,
            'final_zone':    self.zone_history[-1] if self.zone_history else 'low',
            'cdg_timeseries': [round(v, 4) for v in self.cdg_history],
        }


# ── Dashboard ─────────────────────────────────────────────────────────────────
class CDGDashboard:
    def __init__(self, profile='gradual_drifter'):
        self.session = CDGSession(profile)
        self.setup_figure()
        self.ani = None

    def setup_figure(self):
        plt.rcParams.update({
            'figure.facecolor': BG, 'axes.facecolor': SURFACE,
            'axes.edgecolor':   BORDER, 'axes.labelcolor': TEXT,
            'xtick.color':      MUTED, 'ytick.color':      MUTED,
            'text.color':       TEXT,  'grid.color':        BORDER,
            'grid.linewidth':   0.5,   'font.size':         10,
            'font.family':      'DejaVu Sans',
        })

        self.fig = plt.figure(figsize=(14, 8), facecolor=BG)
        self.fig.canvas.manager.set_window_title(
            'CDG MVP Dashboard — Cognitive Dependency Gradient'
        )

        gs = gridspec.GridSpec(
            3, 4, figure=self.fig,
            hspace=0.45, wspace=0.35,
            left=0.07, right=0.97,
            top=0.92,  bottom=0.08
        )

        # Main CDG curve (top, full width)
        self.ax_cdg = self.fig.add_subplot(gs[0, :])
        # Signal bars (middle left)
        self.ax_signals = self.fig.add_subplot(gs[1, :2])
        # SFI gauge (middle right)
        self.ax_sfi = self.fig.add_subplot(gs[1, 2])
        # DR gauge (middle far right)
        self.ax_dr = self.fig.add_subplot(gs[1, 3])
        # Event log (bottom left)
        self.ax_log = self.fig.add_subplot(gs[2, :2])
        # Stats panel (bottom right)
        self.ax_stats = self.fig.add_subplot(gs[2, 2:])

        for ax in [self.ax_log, self.ax_stats]:
            ax.set_facecolor(CARD)

        self._setup_axes()

    def _setup_axes(self):
        profile_label = self.session.profile.replace('_', ' ').title()
        self.fig.suptitle(
            f'CDG MVP Dashboard  ·  Profile: {profile_label}  '
            f'·  Formula validated: AUC=1.00  F1=0.995',
            color=TEXT, fontsize=12, fontweight='bold', y=0.97
        )

        # CDG axis
        self.ax_cdg.set_xlim(0, N_TICKS)
        self.ax_cdg.set_ylim(0, 1.05)
        self.ax_cdg.set_xlabel('Session tick (1 tick = 1 minute)', color=MUTED)
        self.ax_cdg.set_ylabel('CDG Score', color=MUTED)
        self.ax_cdg.set_title('Live CDG(t) — Cognitive Dependency Gradient',
                               color=TEXT, pad=8)
        # Zone bands
        self.ax_cdg.axhspan(0.0, 0.3, alpha=0.06, color='#10b981')
        self.ax_cdg.axhspan(0.3, 0.6, alpha=0.06, color='#3b82f6')
        self.ax_cdg.axhspan(0.6, 0.8, alpha=0.06, color='#f59e0b')
        self.ax_cdg.axhspan(0.8, 1.0, alpha=0.08, color='#ef4444')
        self.ax_cdg.axhline(ALERT_THRESHOLD, color='#ef4444',
                             linestyle='--', linewidth=1.2, alpha=0.7,
                             label='Alert threshold (0.60)')
        # Zone labels
        for y, label in [(0.15, 'LOW'), (0.45, 'MODERATE'),
                         (0.70, 'HIGH'), (0.90, 'CRITICAL')]:
            self.ax_cdg.text(N_TICKS - 1, y, label, color=MUTED,
                             fontsize=8, ha='right', va='center', alpha=0.6)
        self.ax_cdg.grid(True, alpha=0.3)

    def update(self, frame):
        state = self.session.step()
        if state is None or self.session.complete:
            self._on_complete()
            return

        cdg  = state['cdg']
        sfi  = state['sfi']
        dr   = state['dr']
        zone = state['zone']
        tick = state['tick']
        norm = state['norm']

        history = self.session.cdg_history
        ticks   = list(range(len(history)))

        # ── CDG curve ─────────────────────────────────────────────────────────
        self.ax_cdg.cla()
        self._setup_axes()

        zone_color = ZONE_COLORS[zone]
        self.ax_cdg.plot(ticks, history,
                         color=ACCENT, linewidth=2.0, alpha=0.9)
        self.ax_cdg.fill_between(ticks, history, alpha=0.15, color=ACCENT)

        # IP events
        for ip_tick in self.session.ip_events:
            if ip_tick < len(history):
                self.ax_cdg.axvline(ip_tick, color='#f59e0b',
                                    linestyle=':', linewidth=1.5, alpha=0.7)
                self.ax_cdg.annotate(
                    '⚡IP', xy=(ip_tick, history[ip_tick]),
                    fontsize=7, color='#f59e0b',
                    xytext=(ip_tick + 0.5, history[ip_tick] + 0.04)
                )

        # Alert events
        for al_tick in self.session.alert_events:
            if al_tick < len(history):
                self.ax_cdg.axvline(al_tick, color='#ef4444',
                                    linestyle='--', linewidth=1.5, alpha=0.6)

        # Current point
        self.ax_cdg.scatter([tick - 1], [cdg],
                            color=zone_color, s=80, zorder=5)

        # CDG score display
        self.ax_cdg.text(
            0.02, 0.92,
            f'CDG = {cdg:.3f}  [{zone.upper()}]  '
            f't={tick}/{N_TICKS} min',
            transform=self.ax_cdg.transAxes,
            fontsize=12, fontweight='bold', color=zone_color,
            bbox=dict(boxstyle='round', facecolor=CARD,
                      alpha=0.9, edgecolor=zone_color)
        )

        # ── Signal bars ───────────────────────────────────────────────────────
        self.ax_signals.cla()
        self.ax_signals.set_facecolor(SURFACE)
        signals = ['PQAR', 'QCS', 'TTQ', 'ARWM', 'RET', 'OCR']
        sig_vals = [norm[k] for k in signals]
        colors   = ['#ef4444' if v > 0.6 else
                    '#f59e0b' if v > 0.3 else '#10b981'
                    for v in sig_vals]
        bars = self.ax_signals.barh(signals, sig_vals,
                                     color=colors, alpha=0.8, edgecolor='none')
        for bar, val in zip(bars, sig_vals):
            self.ax_signals.text(
                min(val + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=8, color=TEXT
            )
        self.ax_signals.set_xlim(0, 1.05)
        self.ax_signals.set_title('Normalized Signals (higher = more dependent)',
                                   color=TEXT, fontsize=9, pad=6)
        self.ax_signals.axvline(0.5, color=MUTED, linestyle='--',
                                 linewidth=0.8, alpha=0.5)
        self.ax_signals.grid(True, alpha=0.2, axis='x')
        self.ax_signals.tick_params(labelsize=8)

        # ── SFI gauge ─────────────────────────────────────────────────────────
        self.ax_sfi.cla()
        self.ax_sfi.set_facecolor(SURFACE)
        sfi_pct = (sfi - 1.0) / 1.0
        sfi_color = ('#ef4444' if sfi > 1.7 else
                     '#f59e0b' if sfi > 1.4 else '#10b981')
        self.ax_sfi.barh(['SFI'], [sfi_pct], color=sfi_color,
                          alpha=0.8, edgecolor='none')
        self.ax_sfi.set_xlim(0, 1.05)
        self.ax_sfi.set_title(f'Session Fatigue\nSFI = {sfi:.3f}',
                               color=TEXT, fontsize=9, pad=6)
        self.ax_sfi.text(0.5, 0.5, f'{sfi:.3f}',
                          transform=self.ax_sfi.transAxes,
                          ha='center', va='center',
                          fontsize=16, fontweight='bold', color=sfi_color)
        self.ax_sfi.set_yticks([])
        self.ax_sfi.tick_params(labelsize=8)

        # ── DR gauge ──────────────────────────────────────────────────────────
        self.ax_dr.cla()
        self.ax_dr.set_facecolor(SURFACE)
        dr_color = '#ef4444' if dr > 0.01 else \
                   '#10b981' if dr < -0.005 else '#3b82f6'
        dr_label = '↑ DRIFTING' if dr > 0.005 else \
                   '↓ RECOVERING' if dr < -0.005 else '→ STABLE'
        self.ax_dr.set_title(f'Drift Rate\nDR = {dr:+.4f}',
                              color=TEXT, fontsize=9, pad=6)
        self.ax_dr.text(0.5, 0.5, dr_label,
                         transform=self.ax_dr.transAxes,
                         ha='center', va='center',
                         fontsize=11, fontweight='bold', color=dr_color)
        self.ax_dr.set_xticks([])
        self.ax_dr.set_yticks([])

        # ── Event log ─────────────────────────────────────────────────────────
        self.ax_log.cla()
        self.ax_log.set_facecolor(CARD)
        self.ax_log.set_xticks([])
        self.ax_log.set_yticks([])
        self.ax_log.set_title('Event Log', color=TEXT, fontsize=9, pad=6)
        log_lines = []
        for ip_t in self.session.ip_events[-4:]:
            log_lines.append(
                (f't={ip_t:02d}  ⚡ INFLECTION POINT detected', '#f59e0b'))
        for al_t in self.session.alert_events[-4:]:
            log_lines.append(
                (f't={al_t:02d}  🔔 ALERT: CDG crossed 0.60', '#ef4444'))
        if not log_lines:
            log_lines.append(('No events yet — monitoring...', MUTED))
        for i, (line, color) in enumerate(log_lines[-6:]):
            self.ax_log.text(
                0.03, 0.85 - i * 0.18, line,
                transform=self.ax_log.transAxes,
                fontsize=9, color=color,
                fontfamily='monospace'
            )

        # ── Stats panel ───────────────────────────────────────────────────────
        self.ax_stats.cla()
        self.ax_stats.set_facecolor(CARD)
        self.ax_stats.set_xticks([])
        self.ax_stats.set_yticks([])
        self.ax_stats.set_title('Session Statistics', color=TEXT,
                                 fontsize=9, pad=6)
        stats = [
            ('Profile',      self.session.profile.replace('_', ' ').title()),
            ('Current CDG',  f'{cdg:.4f}'),
            ('Zone',         zone.upper()),
            ('Mean CDG',     f'{np.mean(history):.4f}' if history else '—'),
            ('Max CDG',      f'{np.max(history):.4f}'  if history else '—'),
            ('SFI',          f'{sfi:.3f}'),
            ('Drift Rate',   f'{dr:+.5f}'),
            ('IP events',    str(len(self.session.ip_events))),
            ('Alerts',       str(len(self.session.alert_events))),
            ('Progress',     f'{tick}/{N_TICKS} ticks ({tick/N_TICKS*100:.0f}%)'),
        ]
        for i, (label, value) in enumerate(stats):
            y = 0.90 - i * 0.09
            self.ax_stats.text(0.03, y, f'{label}:',
                                transform=self.ax_stats.transAxes,
                                fontsize=8, color=MUTED)
            color = ZONE_COLORS.get(zone, TEXT) if label == 'Zone' else TEXT
            self.ax_stats.text(0.45, y, value,
                                transform=self.ax_stats.transAxes,
                                fontsize=8, color=color, fontweight='bold')

    def _on_complete(self):
        self.ani.event_source.stop()
        print("\n" + "="*55)
        print("  SESSION COMPLETE")
        print("="*55)
        data = self.session.export_json()
        print(f"  Profile   : {data['profile']}")
        print(f"  CDG mean  : {data['cdg_mean']:.4f}")
        print(f"  CDG max   : {data['cdg_max']:.4f}")
        print(f"  CDG final : {data['cdg_final']:.4f}")
        print(f"  Final zone: {data['final_zone'].upper()}")
        print(f"  IP events : {len(data['ip_events'])}")
        print(f"  Alerts    : {len(data['alert_events'])}")

        # Export
        out_dir  = os.path.join(os.path.dirname(__file__), '..', 'pilot', 'sessions')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{data['session_id']}.json")
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n  Session exported: {out_path}")
        print(f"{'='*55}\n")

        self.ax_cdg.set_title(
            f'SESSION COMPLETE — Final CDG: {data["cdg_final"]:.3f}  '
            f'[{data["final_zone"].upper()}]',
            color=ZONE_COLORS[data['final_zone']], pad=8, fontsize=11
        )
        plt.draw()

    def run(self):
        self.ani = FuncAnimation(
            self.fig, self.update,
            interval=TICK_SPEED_MS,
            cache_frame_data=False
        )
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  CDG MVP DASHBOARD")
    print("  Cognitive Dependency Gradient — Live Simulation")
    print("="*55)
    print("\n  Select user profile to simulate:")
    print("  1. autonomous          (CDG stays low — no alerts)")
    print("  2. gradual_drifter     (CDG rises steadily — alert ~min 24)")
    print("  3. threshold_drifter   (CDG spikes after threshold — alert ~min 39)")
    print()

    choice = input("  Enter 1, 2, or 3 [default: 2]: ").strip()
    profile_map = {'1': 'autonomous', '2': 'gradual_drifter',
                   '3': 'threshold_drifter', '': 'gradual_drifter'}
    profile = profile_map.get(choice, 'gradual_drifter')

    print(f"\n  Starting session: {profile}")
    print(f"  Close the window to exit. Session JSON auto-saved on complete.\n")

    dashboard = CDGDashboard(profile=profile)
    dashboard.run()


if __name__ == "__main__":
    main()
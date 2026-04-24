import io
import math
import os
import re

import numpy as np

mpl_config_dir = os.path.join(os.path.dirname(__file__), ".mplconfig")
os.makedirs(mpl_config_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio.v2 as imageio

with open("pluto.log", "r") as f:
    lines = f.readlines()

# Find all analysis section start lines
header_pattern = re.compile(r"^=== Analysis: t = ([\d.eE+-]+) ===$")

times = []
all_r = []
all_rho = []
all_vx1 = []
all_prs = []
all_temp = []
all_nHI = []
all_photon = []
all_grid_photon = []
all_photon_en = []
all_gas_en = []
all_rad_force = []

# Per-snapshot accumulated diagnostics
all_E_ph_emit = []
all_E_ph_esc = []

diag_line1 = re.compile(r"Accumulated.*dM=\s*([\d.eE+-]+).*dE=\s*([\d.eE+-]+).*dP=\s*([\d.eE+-]+)")
diag_line2 = re.compile(r"E_ph_emit=\s*([\d.eE+-]+).*E_ph_esc=\s*([\d.eE+-]+)")

i = 0
while i < len(lines):
    m = header_pattern.match(lines[i].strip())
    if m:
        t = float(m.group(1))
        times.append(t)
        i += 1

        # Parse diagnostic lines
        E_emit, E_esc = 0.0, 0.0
        while i < len(lines) and not lines[i].strip().startswith("Cell"):
            m2 = diag_line2.search(lines[i])
            if m2:
                E_emit = float(m2.group(1))
                E_esc = float(m2.group(2))
            i += 1
        all_E_ph_emit.append(E_emit)
        all_E_ph_esc.append(E_esc)

        # Skip column header and separator lines
        i += 2

        r, rho, vx1, prs, temp, nHI, photon, grid_photon, photon_en, gas_en, rad_force = (
            [], [], [], [], [], [], [], [], [], [], [])
        while i < len(lines) and "|" in lines[i]:
            parts = lines[i].split("|")
            r.append(float(parts[1].strip()))
            rho.append(float(parts[2].strip()))
            vx1.append(float(parts[3].strip()))
            prs.append(float(parts[4].strip()))
            temp.append(float(parts[5].strip()))
            nHI.append(float(parts[6].strip()))
            photon.append(float(parts[7].strip()))
            grid_photon.append(float(parts[8].strip()))
            photon_en.append(float(parts[9].strip()))
            gas_en.append(float(parts[10].strip()))
            rad_force.append(float(parts[11].strip()))
            i += 1

        all_r.append(r)
        all_rho.append(rho)
        all_vx1.append(vx1)
        all_prs.append(prs)
        all_temp.append(temp)
        all_nHI.append(nHI)
        all_photon.append(photon)
        all_grid_photon.append(grid_photon)
        all_photon_en.append(photon_en)
        all_gas_en.append(gas_en)
        all_rad_force.append(rad_force)
    else:
        i += 1

# Convert to numpy arrays
times = np.array(times)
all_r = np.array(all_r)      # shape (n_snapshots, n_cells)
all_rho = np.array(all_rho)
all_vx1 = np.array(all_vx1)
all_prs = np.array(all_prs)
all_temp = np.array(all_temp)
all_nHI = np.array(all_nHI)
all_photon = np.array(all_photon)
all_grid_photon = np.array(all_grid_photon)
all_photon_en = np.array(all_photon_en)
all_gas_en = np.array(all_gas_en)
all_rad_force = np.array(all_rad_force)
all_E_ph_emit = np.array(all_E_ph_emit)
all_E_ph_esc = np.array(all_E_ph_esc)

print(f"Parsed {len(times)} analysis snapshots")
print(f"Time range: {times[0]:.3e} to {times[-1]:.3e} s")
print(f"Cells per snapshot: {all_r.shape[1]}")

# Physical constants
m_H = 1.6726e-24  # proton mass [g]
k_B = 1.3807e-16  # Boltzmann constant [erg/K]
pc = 3.086e18     # parsec [cm]

# Unit conversions
all_r_pc  = all_r / pc                        # cm -> pc
all_n     = all_rho / m_H                     # g/cm³ -> 1/cm³ (number density)
all_v_kms = all_vx1 / 1e5                     # cm/s -> km/s
all_PkB   = all_prs / k_B                     # dyne/cm² -> K/cm³
# Temperature is now read directly from log, no need to compute

# Integrated energy quantities per snapshot (from per-cell values in log)
total_gas_en     = np.sum(all_gas_en, axis=1)       # erg
total_photon_en  = np.sum(all_photon_en, axis=1)    # erg

# Energy conservation: absorbed energy should equal gas energy change
#   E_absorbed = E_ph_emit - E_ph_esc - E_photons_in_box
#   delta_gas  = total_gas_en(t) - total_gas_en(t=0)
E_absorbed = all_E_ph_emit - all_E_ph_esc - total_photon_en
delta_gas  = total_gas_en - total_gas_en[0]

times_yr = times / 3.156e7  # for time-series x-axis

r_pc = all_r_pc[0]  # r grid is the same for all snapshots

TARGET_VIDEO_RUNTIME_S = 60


def positive_floor(values, default):
    values = np.asarray(values)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return default
    return max(np.percentile(positive, 1), default)


def positive_ceiling(values, floor, scale=2.0):
    values = np.asarray(values)
    vmax = np.max(values[np.isfinite(values)], initial=0.0)
    return max(vmax * scale, floor * 10)


def clipped_for_log(values, floor):
    return np.clip(values, floor, None)


def video_fps(n_frames, target_runtime_s=TARGET_VIDEO_RUNTIME_S):
    return max(1, math.ceil(n_frames / target_runtime_s))


def save_video(fig, update_fn, filename, n_frames, dpi):
    fps = video_fps(n_frames)
    with imageio.get_writer(
        filename,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
    ) as writer:
        for frame_idx in range(n_frames):
            update_fn(frame_idx)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
            buf.seek(0)
            writer.append_data(imageio.imread(buf))
            buf.close()
    print(f"Saved {filename} ({fps} fps, ~{n_frames / fps:.0f}s)")


# Compute global y-limits
n_floor = positive_floor(all_n, 1e-30)
n_max = positive_ceiling(all_n, n_floor)
PkB_floor = positive_floor(all_PkB, 1e-30)
PkB_max = positive_ceiling(all_PkB, PkB_floor)
temp_floor = positive_floor(all_temp, 1.0)
temp_max = positive_ceiling(all_temp, temp_floor)
nHI_floor = positive_floor(all_nHI, 1e-30)
nHI_max = positive_ceiling(all_nHI, nHI_floor)
speed_floor = positive_floor(np.abs(all_v_kms), 1e-2)
speed_max = positive_ceiling(np.abs(all_v_kms), speed_floor, scale=1.2)
photon_floor = positive_floor(np.concatenate((all_photon.ravel(), all_grid_photon.ravel())), 0.1)
photon_max = positive_ceiling(np.concatenate((all_photon.ravel(), all_grid_photon.ravel())), photon_floor)
photon_en_floor = positive_floor(all_photon_en, 1e-30)
photon_en_max = positive_ceiling(all_photon_en, photon_en_floor)
rad_force_abs = np.abs(all_rad_force)
rf_floor = positive_floor(rad_force_abs, 1e-30)
rf_max = positive_ceiling(rad_force_abs, rf_floor)

all_n_plot = clipped_for_log(all_n, n_floor)
all_speed_plot = clipped_for_log(np.abs(all_v_kms), speed_floor)
all_PkB_plot = clipped_for_log(all_PkB, PkB_floor)
all_temp_plot = clipped_for_log(all_temp, temp_floor)
all_nHI_plot = clipped_for_log(all_nHI, nHI_floor)
all_photon_plot = clipped_for_log(all_photon, photon_floor)
all_grid_photon_plot = clipped_for_log(all_grid_photon, photon_floor)
all_photon_en_plot = clipped_for_log(all_photon_en, photon_en_floor)
all_rad_force_plot = clipped_for_log(rad_force_abs, rf_floor)

# ===== Figure 1: Radial profiles (8 panels, 2x4) =====
fig1, axes1 = plt.subplots(2, 4, figsize=(20, 9))
for row in range(2):
    for col in range(4):
        axes1[row, col].sharex(axes1[0, 0])
fig1.set_facecolor("white")

line_rho, = axes1[0, 0].plot(r_pc, all_n_plot[0], "b-")
line_vx1, = axes1[0, 1].plot(r_pc, all_speed_plot[0], "b-")
line_prs, = axes1[0, 2].plot(r_pc, all_PkB_plot[0], "b-")
line_temp, = axes1[0, 3].plot(r_pc, all_temp_plot[0], "b-")
line_nHI, = axes1[1, 0].plot(r_pc, all_nHI_plot[0], "b-")
line_photon, = axes1[1, 1].plot(r_pc, all_photon_plot[0], "b-", label="All")
line_grid_photon, = axes1[1, 1].plot(r_pc, all_grid_photon_plot[0], "r--", label="Grid-emitted")
line_photon_en, = axes1[1, 2].plot(r_pc, all_photon_en_plot[0], "b-")
line_rad_force, = axes1[1, 3].plot(r_pc, all_rad_force_plot[0], "b-")

axes1[0, 0].set_ylabel(r"Density ($n_p$ / cm$^{-3}$)")
axes1[0, 0].set_yscale("log")
axes1[0, 0].set_xlim(r_pc[0], r_pc[-1])
axes1[0, 0].set_ylim(n_floor, n_max)

axes1[0, 1].set_ylabel("Speed (km/s)")
axes1[0, 1].set_yscale("log")
axes1[0, 1].set_ylim(speed_floor, speed_max)

axes1[0, 2].set_ylabel(r"P / $k_B$ (K cm$^{-3}$)")
axes1[0, 2].set_yscale("log")
axes1[0, 2].set_ylim(PkB_floor, PkB_max)

axes1[0, 3].set_ylabel("Temperature (K)")
axes1[0, 3].set_yscale("log")
axes1[0, 3].set_ylim(temp_floor, temp_max)

axes1[1, 0].set_ylabel(r"$n_{\mathrm{HI}}$ (cm$^{-3}$)")
axes1[1, 0].set_yscale("log")
axes1[1, 0].set_ylim(nHI_floor, nHI_max)
axes1[1, 0].set_xlabel("r (pc)")

axes1[1, 1].set_ylabel("Photon Count")
axes1[1, 1].set_yscale("log")
axes1[1, 1].set_ylim(photon_floor, photon_max)
axes1[1, 1].set_xlabel("r (pc)")
axes1[1, 1].legend(fontsize=8)

axes1[1, 2].set_ylabel("Photon Energy (erg)")
axes1[1, 2].set_yscale("log")
axes1[1, 2].set_ylim(photon_en_floor, photon_en_max)
axes1[1, 2].set_xlabel("r (pc)")

axes1[1, 3].set_ylabel(r"|Rad. Force| (dyn/cm$^3$)")
axes1[1, 3].set_yscale("log")
axes1[1, 3].set_ylim(rf_floor, rf_max)
axes1[1, 3].set_xlabel("r (pc)")

title1 = fig1.suptitle("", fontsize=14, fontweight="bold")
time_text1 = axes1[0, 3].text(
    0.98, 0.95, "", transform=axes1[0, 3].transAxes,
    ha="right", va="top", fontsize=11,
    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8),
)
fig1.tight_layout()

def update_profiles(frame):
    line_rho.set_data(r_pc, all_n_plot[frame])
    line_vx1.set_data(r_pc, all_speed_plot[frame])
    line_prs.set_data(r_pc, all_PkB_plot[frame])
    line_temp.set_data(r_pc, all_temp_plot[frame])
    line_nHI.set_data(r_pc, all_nHI_plot[frame])
    line_photon.set_data(r_pc, all_photon_plot[frame])
    line_grid_photon.set_data(r_pc, all_grid_photon_plot[frame])
    line_photon_en.set_data(r_pc, all_photon_en_plot[frame])
    line_rad_force.set_data(r_pc, all_rad_force_plot[frame])
    t_sec = times[frame]
    t_yr = t_sec / 3.156e7
    title1.set_text(f"Radial Profiles — t = {t_sec:.3e} s")
    time_text1.set_text(f"{t_yr:.1f} yr")
    return (line_rho, line_vx1, line_prs, line_temp, line_nHI, line_photon,
            line_grid_photon, line_photon_en, line_rad_force, title1, time_text1)

# Save radial profiles mp4, capped at ~60 s runtime
save_video(fig1, update_profiles, "radial_profiles.mp4", len(times), dpi=96)

# ===== Figure 2: Energy conservation panel =====
fig2, ax2 = plt.subplots(figsize=(8, 4.8))
fig2.set_facecolor("white")

energy_abs_mask = (times_yr > 0) & (E_absorbed > 0)
energy_gas_mask = (times_yr > 0) & (delta_gas > 0)
positive_time = times_yr[times_yr > 0]
positive_energy = np.concatenate((E_absorbed[energy_abs_mask], delta_gas[energy_gas_mask]))
energy_ymin = 0.8 * positive_energy.min()
energy_ymax = 1.25 * positive_energy.max()

ax2.plot(
    times_yr[energy_abs_mask], E_absorbed[energy_abs_mask], "b-", lw=1,
    label=r"$E_{\rm emit} - E_{\rm esc} - E_{\rm ph,box}$",
)
ax2.plot(
    times_yr[energy_gas_mask], delta_gas[energy_gas_mask], "r--", lw=1,
    label=r"$\Delta E_{\rm gas}$",
)
conserv_marker_abs, = ax2.plot(times_yr[0], E_absorbed[0], "bo", ms=6)
conserv_marker_gas, = ax2.plot(times_yr[0], delta_gas[0], "rs", ms=5)
ax2.set_ylabel("Energy (erg)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Time (yr)")
ax2.set_xlim(positive_time[0], positive_time[-1])
ax2.set_ylim(energy_ymin, energy_ymax)
ax2.legend(fontsize=8)

title2 = fig2.suptitle("", fontsize=14, fontweight="bold")
fig2.tight_layout()
title2.set_text("Energy Conservation")
conserv_marker_abs.set_visible(False)
conserv_marker_gas.set_visible(False)
fig2.savefig("energy_profiles.png", dpi=150, facecolor="white")
print("Saved energy_profiles.png")

# Interactive display with pause support
paused = False

def on_key(event):
    global paused
    if event.key == " ":
        if paused:
            anim1.resume()
        else:
            anim1.pause()
        paused = not paused

fig1.canvas.mpl_connect("key_press_event", on_key)

backend = plt.get_backend().lower()
if "agg" not in backend:
    anim1 = FuncAnimation(fig1, update_profiles, frames=len(times), interval=750, blit=False, repeat=True)
    print("Press SPACE to pause/resume")
    plt.show()
else:
    print("Skipping interactive display because the current Matplotlib backend is non-interactive.")

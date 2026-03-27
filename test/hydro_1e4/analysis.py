import re
import numpy as np
import matplotlib.pyplot as plt

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
mu = 1.2          # mean molecular weight
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

# --- Animated radial profiles (non-energy) ---
from matplotlib.animation import FuncAnimation
import imageio
import io

r_pc = all_r_pc[0]  # r grid is the same for all snapshots

# Compute global y-limits
n_min, n_max = all_n.min(), all_n.max()
PkB_min, PkB_max = all_PkB.min(), all_PkB.max()
temp_min, temp_max = all_temp.min(), all_temp.max()
nHI_min, nHI_max = all_nHI.min(), all_nHI.max()
photon_min, photon_max = 0, 1000

# ===== Figure 1: Radial profiles (8 panels, 2x4) =====
fig1, axes1 = plt.subplots(2, 4, figsize=(20, 9))
for row in range(2):
    for col in range(4):
        axes1[row, col].sharex(axes1[0, 0])
fig1.set_facecolor("white")

line_rho, = axes1[0, 0].plot(r_pc, all_n[0], "b-")
line_vx1, = axes1[0, 1].plot(r_pc, all_v_kms[0], "b-")
line_prs, = axes1[0, 2].plot(r_pc, all_PkB[0], "b-")
line_temp, = axes1[0, 3].plot(r_pc, all_temp[0], "b-")
line_nHI, = axes1[1, 0].plot(r_pc, all_nHI[0], "b-")
line_photon, = axes1[1, 1].plot(r_pc, all_photon[0], "b-", label="All")
line_grid_photon, = axes1[1, 1].plot(r_pc, all_grid_photon[0], "r--", label="Grid-emitted")
line_photon_en, = axes1[1, 2].plot(r_pc, all_photon_en[0], "b-")
line_rad_force, = axes1[1, 3].plot(r_pc, all_rad_force[0], "b-")

axes1[0, 0].set_ylabel(r"Density ($n_p$ / cm$^{-3}$)")
axes1[0, 0].set_yscale("log")
axes1[0, 0].set_xlim(r_pc[0], r_pc[-1])
axes1[0, 0].set_ylim(n_min * 0.5, n_max * 2)

axes1[0, 1].set_ylabel("Velocity (km/s)")
axes1[0, 1].set_yscale("log")
axes1[0, 1].set_ylim(0, 5000)

axes1[0, 2].set_ylabel(r"P / $k_B$ (K cm$^{-3}$)")
axes1[0, 2].set_yscale("log")
axes1[0, 2].set_ylim(PkB_min * 0.5, PkB_max * 2)

axes1[0, 3].set_ylabel("Temperature (K)")
axes1[0, 3].set_yscale("log")
axes1[0, 3].set_ylim(temp_min * 0.5, temp_max * 2)

axes1[1, 0].set_ylabel(r"$n_{\mathrm{HI}}$ (cm$^{-3}$)")
axes1[1, 0].set_yscale("log")
axes1[1, 0].set_ylim(nHI_min * 0.5, nHI_max * 2)
axes1[1, 0].set_xlabel("r (pc)")

axes1[1, 1].set_ylabel("Photon Count")
axes1[1, 1].set_yscale("log")
axes1[1, 1].set_ylim(max(photon_min * 0.5, 0.1), photon_max * 2)
axes1[1, 1].set_xlabel("r (pc)")
axes1[1, 1].legend(fontsize=8)

photon_en_min = all_photon_en[all_photon_en > 0].min() if np.any(all_photon_en > 0) else 1e-10
photon_en_max = all_photon_en.max() if all_photon_en.max() > 0 else 1.0
axes1[1, 2].set_ylabel("Photon Energy (erg)")
axes1[1, 2].set_yscale("log")
axes1[1, 2].set_ylim(photon_en_min * 0.5, photon_en_max * 2)
axes1[1, 2].set_xlabel("r (pc)")

rad_force_abs = np.abs(all_rad_force)
rf_min = rad_force_abs[rad_force_abs > 0].min() if np.any(rad_force_abs > 0) else 1e-30
rf_max = rad_force_abs.max() if rad_force_abs.max() > 0 else 1.0
axes1[1, 3].set_ylabel(r"Rad. Force (dyn/cm$^3$)")
axes1[1, 3].set_yscale("log")
axes1[1, 3].set_ylim(rf_min * 0.5, rf_max * 2)
axes1[1, 3].set_xlabel("r (pc)")

title1 = fig1.suptitle("", fontsize=14, fontweight="bold")
time_text1 = axes1[0, 3].text(
    0.98, 0.95, "", transform=axes1[0, 3].transAxes,
    ha="right", va="top", fontsize=11,
    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8),
)
fig1.tight_layout()

def update_profiles(frame):
    line_rho.set_data(r_pc, all_n[frame])
    line_vx1.set_data(r_pc, all_v_kms[frame])
    line_prs.set_data(r_pc, all_PkB[frame])
    line_temp.set_data(r_pc, all_temp[frame])
    line_nHI.set_data(r_pc, all_nHI[frame])
    line_photon.set_data(r_pc, all_photon[frame])
    line_grid_photon.set_data(r_pc, all_grid_photon[frame])
    line_photon_en.set_data(r_pc, all_photon_en[frame])
    line_rad_force.set_data(r_pc, np.abs(all_rad_force[frame]))
    t_sec = times[frame]
    t_yr = t_sec / 3.156e7
    title1.set_text(f"Radial Profiles — t = {t_sec:.3e} s")
    time_text1.set_text(f"{t_yr:.1f} yr")
    return (line_rho, line_vx1, line_prs, line_temp, line_nHI, line_photon,
            line_grid_photon, line_photon_en, line_rad_force, title1, time_text1)

# Save radial profiles mp4
frames_prof = []
for frame_idx in range(len(times)):
    update_profiles(frame_idx)
    buf = io.BytesIO()
    fig1.savefig(buf, format="png", dpi=150, facecolor="white")
    buf.seek(0)
    frames_prof.append(imageio.imread(buf))
    buf.close()

imageio.mimwrite("radial_profiles.mp4", frames_prof, fps=2, codec="libx264",
                 output_params=["-pix_fmt", "yuv420p"])
print("Saved radial_profiles.mp4")

# ===== Figure 2: Energy panels (3 panels, 1x3) =====
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.set_facecolor("white")

# Panel 0: Total gas energy vs time
axes2[0].plot(times_yr, total_gas_en, "k-", lw=1)
gas_en_marker, = axes2[0].plot(times_yr[0], total_gas_en[0], "ro", ms=6)
axes2[0].set_ylabel("Total Gas Energy (erg)")
axes2[0].set_yscale("log")
axes2[0].set_xlabel("Time (yr)")
axes2[0].set_xlim(times_yr[0], times_yr[-1])

# Panel 1: Total injected photon energy vs time
axes2[1].plot(times_yr, all_E_ph_emit, "k-", lw=1)
emit_marker, = axes2[1].plot(times_yr[0], all_E_ph_emit[0], "ro", ms=6)
axes2[1].set_ylabel(r"$E_{\rm ph,emit}$ (erg)")
axes2[1].set_yscale("log")
axes2[1].set_xlabel("Time (yr)")
axes2[1].set_xlim(times_yr[0], times_yr[-1])

# Panel 2: Energy conservation check
axes2[2].plot(times_yr, E_absorbed, "b-", lw=1, label=r"$E_{\rm emit} - E_{\rm esc} - E_{\rm ph,box}$")
axes2[2].plot(times_yr, delta_gas, "r--", lw=1, label=r"$\Delta E_{\rm gas}$")
conserv_marker_abs, = axes2[2].plot(times_yr[0], E_absorbed[0], "bo", ms=6)
conserv_marker_gas, = axes2[2].plot(times_yr[0], delta_gas[0], "rs", ms=5)
axes2[2].set_ylabel("Energy (erg)")
axes2[2].set_yscale("symlog", linthresh=1e30)
axes2[2].set_xlabel("Time (yr)")
axes2[2].set_xlim(times_yr[0], times_yr[-1])
axes2[2].legend(fontsize=8)

axes2[2].set_ylim(9e44, 2e48)

title2 = fig2.suptitle("", fontsize=14, fontweight="bold")
fig2.tight_layout()

def update_energy(frame):
    gas_en_marker.set_data([times_yr[frame]], [total_gas_en[frame]])
    emit_marker.set_data([times_yr[frame]], [all_E_ph_emit[frame]])
    conserv_marker_abs.set_data([times_yr[frame]], [E_absorbed[frame]])
    conserv_marker_gas.set_data([times_yr[frame]], [delta_gas[frame]])
    t_sec = times[frame]
    t_yr = t_sec / 3.156e7
    title2.set_text(f"Energy — t = {t_sec:.3e} s  ({t_yr:.1f} yr)")
    return (gas_en_marker, emit_marker, conserv_marker_abs, conserv_marker_gas,
            title2)

# Save energy mp4
frames_en = []
for frame_idx in range(len(times)):
    update_energy(frame_idx)
    buf = io.BytesIO()
    fig2.savefig(buf, format="png", dpi=150, facecolor="white")
    buf.seek(0)
    frames_en.append(imageio.imread(buf))
    buf.close()

imageio.mimwrite("energy_profiles.mp4", frames_en, fps=2, codec="libx264",
                 output_params=["-pix_fmt", "yuv420p"])
print("Saved energy_profiles.mp4")

# Interactive display with pause support
paused = False

def on_key(event):
    global paused
    if event.key == " ":
        if paused:
            anim1.resume()
            anim2.resume()
        else:
            anim1.pause()
            anim2.pause()
        paused = not paused

fig1.canvas.mpl_connect("key_press_event", on_key)
fig2.canvas.mpl_connect("key_press_event", on_key)

anim1 = FuncAnimation(fig1, update_profiles, frames=len(times), interval=750, blit=False, repeat=True)
anim2 = FuncAnimation(fig2, update_energy, frames=len(times), interval=750, blit=False, repeat=True)

print("Press SPACE to pause/resume")
plt.show()

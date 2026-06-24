#!/usr/bin/env python3
"""
Reachable workspace visualizer
Usage: python3 visualize_workspace.py <csv_path>
"""
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "eef_track.csv"

# ── Load data ───────────────────────────────────────────────
ok, ng = [], []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        if row["ik"] not in ("OK", "NG"):
            continue
        pt = (float(row["x"]), float(row["y"]), float(row["z"]))
        (ok if row["ik"] == "OK" else ng).append(pt)

ok = np.array(ok)
ng = np.array(ng)
# Limit to scan range
ok = ok[(ok[:, 0] >= -0.505) & (ok[:, 0] <= 0.505)]
ng = ng[(ng[:, 0] >= -0.505) & (ng[:, 0] <= 0.505)]

total = len(ok) + len(ng)
print(f"OK: {len(ok)}  NG: {len(ng)}  Rate: {len(ok)/total*100:.1f}%")

# ── Figure 1: 3D scatter ────────────────────────────────────
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(ng[:, 0], ng[:, 1], ng[:, 2], s=4, c="#cccccc", alpha=0.2, label="NG")
ax.scatter(ok[:, 0], ok[:, 1], ok[:, 2], s=8, c="#e03030", alpha=0.7, label="OK")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]  (left=+)"); ax.set_zlabel("Z [m]")
ax.set_title("Right hand reachable workspace (3D)")
ax.invert_yaxis()  # Y: left=positive (robot left)
ax.legend(); ax.view_init(elev=25, azim=135)
plt.tight_layout()
plt.savefig("workspace_3d.png", dpi=150)
print("Saved: workspace_3d.png")
plt.close()

# ── Figure 2: Reachability heatmap X-Y (Z averaged) ────────
step = 0.025
all_pts = np.vstack([ok, ng])
xv = np.arange(-0.5, 0.505, step)
yv = np.arange(-0.5, 0.505, step)
rate_xy = np.full((len(xv), len(yv)), np.nan)
for xi, x0 in enumerate(xv):
    for yi, y0 in enumerate(yv):
        mask = (np.abs(all_pts[:, 0] - x0) < step/2) & \
               (np.abs(all_pts[:, 1] - y0) < step/2)
        ok_m  = (np.abs(ok[:, 0] - x0) < step/2) & \
                (np.abs(ok[:, 1] - y0) < step/2)
        if mask.sum() > 0:
            rate_xy[xi, yi] = ok_m.sum() / mask.sum() * 100

fig, ax = plt.subplots(figsize=(8, 7))
# 鳥瞰図: 横=Y（左が正）、縦=X（上が前方）
im = ax.imshow(rate_xy, origin="lower", aspect="equal",
               extent=[yv[0]-step/2, yv[-1]+step/2,
                       xv[0]-step/2, xv[-1]+step/2],
               cmap="RdYlGn", vmin=0, vmax=100)
ax.invert_xaxis()  # Y: left=positive (robot left side)
plt.colorbar(im, ax=ax, label="Reachability [%]")
ax.axhline(0, color="k", lw=0.5, ls="--"); ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("Y [m]  (left=+)"); ax.set_ylabel("X [m]  (up=forward)")
ax.set_title("Reachability heatmap (X-Y bird's eye, Z averaged)")
plt.tight_layout()
plt.savefig("workspace_heatmap_xy.png", dpi=150)
print("Saved: workspace_heatmap_xy.png")
plt.close()

# ── Figure 3: Reachability heatmap X-Z (Y averaged) ────────
zv = np.arange(0.0, 0.505, step)
rate_xz = np.full((len(xv), len(zv)), np.nan)
for xi, x0 in enumerate(xv):
    for zi, z0 in enumerate(zv):
        mask = (np.abs(all_pts[:, 0] - x0) < step/2) & \
               (np.abs(all_pts[:, 2] - z0) < step/2)
        ok_m  = (np.abs(ok[:, 0] - x0) < step/2) & \
                (np.abs(ok[:, 2] - z0) < step/2)
        if mask.sum() > 0:
            rate_xz[xi, zi] = ok_m.sum() / mask.sum() * 100

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(rate_xz.T, origin="lower", aspect="auto",
               extent=[xv[0]-step/2, xv[-1]+step/2,
                       zv[0]-step/2, zv[-1]+step/2],
               cmap="RdYlGn", vmin=0, vmax=100)
plt.colorbar(im, ax=ax, label="Reachability [%]")
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("X [m]"); ax.set_ylabel("Z [m]")
ax.set_title("Reachability heatmap (X-Z, Y averaged)")
plt.tight_layout()
plt.savefig("workspace_heatmap_xz.png", dpi=150)
print("Saved: workspace_heatmap_xz.png")
plt.close()

# ── Figure 4: X-Y slices per Z ─────────────────────────────
z_levels = np.unique(np.round(all_pts[:, 2], 3))
z_sel = z_levels[::4]
ncols = 4
nrows = int(np.ceil(len(z_sel) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
axes = axes.flatten()
for i, z0 in enumerate(z_sel):
    ax = axes[i]
    ok_s = ok[np.abs(ok[:, 2] - z0) < step/2]
    ng_s = ng[np.abs(ng[:, 2] - z0) < step/2]
    ax.scatter(ng_s[:, 1], ng_s[:, 0], s=8, c="#cccccc", alpha=0.5)
    ax.scatter(ok_s[:, 1], ok_s[:, 0], s=8, c="#e03030", alpha=0.8)
    ax.axhline(0, color="k", lw=0.4, ls="--"); ax.axvline(0, color="k", lw=0.4, ls="--")
    ax.set_title(f"Z={z0:.3f}m", fontsize=9)
    ax.set_xlabel("Y [m] (left=+)"); ax.set_ylabel("X [m] (up=fwd)")
    ax.set_xlim(0.55, -0.55)  # Y: left=positive (robot left)
    ax.set_ylim(-0.55, 0.55)
    ax.grid(True, linewidth=0.3)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("X-Y slices per Z level  (red=OK, gray=NG)", fontsize=11)
plt.tight_layout()
plt.savefig("workspace_slices_z.png", dpi=150)
print("Saved: workspace_slices_z.png")
plt.close()

# ── Figure 5: 3D voxel reachability map ────────────────────
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
zv_full = np.arange(0.0, 0.505, step)
for xi, x0 in enumerate(xv):
    for yi, y0 in enumerate(yv):
        for zi, z0 in enumerate(zv_full):
            mask = (np.abs(all_pts[:, 0] - x0) < step/2) & \
                   (np.abs(all_pts[:, 1] - y0) < step/2) & \
                   (np.abs(all_pts[:, 2] - z0) < step/2)
            ok_m = (np.abs(ok[:, 0] - x0) < step/2) & \
                   (np.abs(ok[:, 1] - y0) < step/2) & \
                   (np.abs(ok[:, 2] - z0) < step/2)
            n_all = mask.sum()
            n_ok  = ok_m.sum()
            if n_all == 0:
                continue
            r = n_ok / n_all  # 0..1
            color = (1 - r, r, 0, 0.6)  # red→green
            ax.scatter(x0, y0, z0, s=20, color=color, marker="s")

# colorbar 代わりの凡例
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=(1,0,0,0.6), label="NG (0%)"),
    Patch(color=(0.5,0.5,0,0.6), label="50%"),
    Patch(color=(0,1,0,0.6), label="OK (100%)"),
], loc="upper left")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]  (left=+)"); ax.set_zlabel("Z [m]")
ax.set_title("Reachability 3D voxel map (red=NG, green=OK)")
ax.invert_yaxis()  # Y: left=positive (robot left)
ax.view_init(elev=20, azim=130)
plt.tight_layout()
plt.savefig("workspace_3d_voxel.png", dpi=150)
print("Saved: workspace_3d_voxel.png")
plt.close()

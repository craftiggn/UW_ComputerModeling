import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Circle
from scipy.stats import linregress
import math
import os

# ==============================================================================
# 1. ARCHITECTURE & CORE SIMULATION (NUMBA)
# ==============================================================================

@njit
def get_admissible_angle(px, py, parent_id, pos_x, pos_y, grid_head, grid_next):
    """
    Physics Engine: Evaluates the 5x5 grid neighborhood to find available 
    angular gaps for placing a new cell at distance 2.0.
    """
    intervals = np.zeros((1000, 2), dtype=np.float64)
    num_intervals = 0

    # Uniform grid assignment using floor to avoid truncation-to-zero artifacts
    gx_p = int(math.floor(px / 2.0)) + 250
    gy_p = int(math.floor(py / 2.0)) + 250

    # Evaluate 5x5 grid neighborhood
    for gx in range(gx_p - 2, gx_p + 3):
        for gy in range(gy_p - 2, gy_p + 3):
            if gx < 0 or gx >= 500 or gy < 0 or gy >= 500:
                continue
            
            curr = grid_head[gx, gy]
            while curr != -1:
                if curr != parent_id:
                    dx = pos_x[curr] - px
                    dy = pos_y[curr] - py
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if 0.0 < dist < 4.0:
                        alpha = np.arccos(dist / 4.0)
                        phi = np.arctan2(dy, dx)
                        if phi < 0.0:
                            phi += 2.0 * np.pi
                        
                        start = phi - alpha
                        end = phi + alpha

                        # Handle circular wrap-around to keep intervals strictly in [0, 2pi)
                        if start < 0.0:
                            intervals[num_intervals, 0] = 0.0
                            intervals[num_intervals, 1] = end
                            num_intervals += 1
                            intervals[num_intervals, 0] = start + 2.0 * np.pi
                            intervals[num_intervals, 1] = 2.0 * np.pi
                            num_intervals += 1
                        elif end > 2.0 * np.pi:
                            intervals[num_intervals, 0] = start
                            intervals[num_intervals, 1] = 2.0 * np.pi
                            num_intervals += 1
                            intervals[num_intervals, 0] = 0.0
                            intervals[num_intervals, 1] = end - 2.0 * np.pi
                            num_intervals += 1
                        else:
                            intervals[num_intervals, 0] = start
                            intervals[num_intervals, 1] = end
                            num_intervals += 1
                curr = grid_next[curr]

    if num_intervals == 0:
        return np.random.rand() * 2.0 * np.pi

    # Manual Bubble Sort by start angle
    for i in range(num_intervals):
        for j in range(0, num_intervals - i - 1):
            if intervals[j, 0] > intervals[j + 1, 0]:
                temp_s = intervals[j, 0]
                temp_e = intervals[j, 1]
                intervals[j, 0] = intervals[j + 1, 0]
                intervals[j, 1] = intervals[j + 1, 1]
                intervals[j + 1, 0] = temp_s
                intervals[j + 1, 1] = temp_e

    # Merge overlapping intervals
    merged = np.zeros((num_intervals, 2), dtype=np.float64)
    m_count = 0
    curr_s = intervals[0, 0]
    curr_e = intervals[0, 1]

    for i in range(1, num_intervals):
        if intervals[i, 0] <= curr_e + 1e-7: # Small epsilon to prevent microscopic false-gaps
            if intervals[i, 1] > curr_e:
                curr_e = intervals[i, 1]
        else:
            merged[m_count, 0] = curr_s
            merged[m_count, 1] = curr_e
            m_count += 1
            curr_s = intervals[i, 0]
            curr_e = intervals[i, 1]

    merged[m_count, 0] = curr_s
    merged[m_count, 1] = curr_e
    m_count += 1

    # Find open gaps
    gaps = np.zeros((m_count + 1, 2), dtype=np.float64)
    g_count = 0

    if merged[0, 0] > 0.0:
        gaps[g_count, 0] = 0.0
        gaps[g_count, 1] = merged[0, 0]
        g_count += 1

    for i in range(m_count - 1):
        gaps[g_count, 0] = merged[i, 1]
        gaps[g_count, 1] = merged[i + 1, 0]
        g_count += 1

    if merged[m_count - 1, 1] < 2.0 * np.pi:
        gaps[g_count, 0] = merged[m_count - 1, 1]
        gaps[g_count, 1] = 2.0 * np.pi
        g_count += 1

    # Sample a uniform random angle from available gaps
    total_gap = 0.0
    for i in range(g_count):
        total_gap += gaps[i, 1] - gaps[i, 0]

    if total_gap < 1e-5:
        return -100.0

    rand_val = np.random.rand() * total_gap
    for i in range(g_count):
        w = gaps[i, 1] - gaps[i, 0]
        if rand_val <= w:
            return gaps[i, 0] + rand_val
        rand_val -= w

    return -100.0


@njit
def run_eden_c_simulation(N_MAX=50000, track_interval=100):
    # Pre-allocate State Arrays
    pos_x = np.zeros(N_MAX, dtype=np.float64)
    pos_y = np.zeros(N_MAX, dtype=np.float64)
    is_living = np.ones(N_MAX, dtype=np.bool_)
    parent_ids = np.full(N_MAX, -1, dtype=np.int32)

    # Grid-Based Spatial Hash
    grid_head = np.full((500, 500), -1, dtype=np.int32)
    grid_next = np.full(N_MAX, -1, dtype=np.int32)

    # O(1) Living Pool Management
    living_indices = np.zeros(N_MAX, dtype=np.int32)
    living_map = np.zeros(N_MAX, dtype=np.int32)

    # Initialize Seed Cell
    pos_x[0], pos_y[0] = 0.0, 0.0
    grid_head[250, 250] = 0
    living_indices[0] = 0
    living_map[0] = 0
    
    num_living = 1
    num_cells = 1

    # Stats Tracking Arrays
    max_tracks = N_MAX // track_interval + 10
    track_N = np.zeros(max_tracks, dtype=np.int32)
    track_L = np.zeros(max_tracks, dtype=np.int32)
    track_R = np.zeros(max_tracks, dtype=np.float64)
    track_sigma = np.zeros(max_tracks, dtype=np.float64)
    track_idx = 0

    # Snapshot Arrays
    snap_x = np.zeros(3000, dtype=np.float64)
    snap_y = np.zeros(3000, dtype=np.float64)
    snap_living = np.zeros(3000, dtype=np.bool_)
    snap_parent = np.zeros(3000, dtype=np.int32)

    while num_cells < N_MAX and num_living > 0:
        rand_idx = np.random.randint(0, num_living)
        parent_cell = living_indices[rand_idx]

        angle = get_admissible_angle(pos_x[parent_cell], pos_y[parent_cell], parent_cell, pos_x, pos_y, grid_head, grid_next)

        if angle == -100.0:
            # Mark dead and O(1) swap-delete from living map
            is_living[parent_cell] = False
            last_cell = living_indices[num_living - 1]
            living_indices[rand_idx] = last_cell
            living_map[last_cell] = rand_idx
            num_living -= 1
        else:
            c_idx = num_cells
            nx = pos_x[parent_cell] + 2.0 * np.cos(angle)
            ny = pos_y[parent_cell] + 2.0 * np.sin(angle)

            pos_x[c_idx] = nx
            pos_y[c_idx] = ny
            parent_ids[c_idx] = parent_cell

            # Insert into spatial hash
            gx = int(math.floor(nx / 2.0)) + 250
            gy = int(math.floor(ny / 2.0)) + 250
            grid_next[c_idx] = grid_head[gx, gy]
            grid_head[gx, gy] = c_idx

            # Add to living pool
            living_indices[num_living] = c_idx
            living_map[c_idx] = num_living
            num_living += 1
            num_cells += 1

            # Routine Statistics Tracking & Snapshot Triggers
            if num_cells % track_interval == 0 or num_cells == 3000:
                
                # Evaluate True Boundary (Backwards loop to safely swap-delete while iterating)
                for k in range(num_living - 1, -1, -1):
                    cell = living_indices[k]
                    t_angle = get_admissible_angle(pos_x[cell], pos_y[cell], cell, pos_x, pos_y, grid_head, grid_next)
                    if t_angle == -100.0:
                        is_living[cell] = False
                        last_c = living_indices[num_living - 1]
                        living_indices[k] = last_c
                        living_map[last_c] = k
                        num_living -= 1

                if num_cells == 3000:
                    snap_x[:] = pos_x[:3000]
                    snap_y[:] = pos_y[:3000]
                    snap_living[:] = is_living[:3000]
                    snap_parent[:] = parent_ids[:3000]

                if num_cells % track_interval == 0:
                    # Calculate Center of Mass (COM)
                    com_x = np.sum(pos_x[:num_cells]) / num_cells
                    com_y = np.sum(pos_y[:num_cells]) / num_cells

                    # Compute True Living Boundary Radius & Thickness
                    sum_r = 0.0
                    for k in range(num_living):
                        cell = living_indices[k]
                        dx = pos_x[cell] - com_x
                        dy = pos_y[cell] - com_y
                        sum_r += np.sqrt(dx*dx + dy*dy)
                    mean_r = sum_r / num_living if num_living > 0 else 0.0

                    sum_sq_diff = 0.0
                    for k in range(num_living):
                        cell = living_indices[k]
                        dx = pos_x[cell] - com_x
                        dy = pos_y[cell] - com_y
                        r = np.sqrt(dx*dx + dy*dy)
                        sum_sq_diff += (r - mean_r)**2
                    std_r = np.sqrt(sum_sq_diff / num_living) if num_living > 0 else 0.0

                    track_N[track_idx] = num_cells
                    track_L[track_idx] = num_living
                    track_R[track_idx] = mean_r
                    track_sigma[track_idx] = std_r
                    track_idx += 1

    return (pos_x, pos_y, parent_ids,
            snap_x, snap_y, snap_living, snap_parent,
            track_N[:track_idx], track_L[:track_idx], track_R[:track_idx], track_sigma[:track_idx])

# ==============================================================================
# 2. MAIN EXECUTION & STATISTICAL VERIFICATION
# ==============================================================================

if __name__ == "__main__":
    print("Running Eden-C Simulation (Numba JIT Compilation overhead may take a moment)...")
    (pos_x, pos_y, parent_ids, 
     snap_x, snap_y, snap_living, snap_parent, 
     track_N, track_L, track_R, track_sigma) = run_eden_c_simulation(N_MAX=50000, track_interval=100)
    print("Simulation Complete. Calculating Statistics...")

    # --- Interior Density (\rho) ---
    com_50k_x = np.mean(pos_x)
    com_50k_y = np.mean(pos_y)
    r2_50k = (pos_x - com_50k_x)**2 + (pos_y - com_50k_y)**2
    r2_sorted = np.sort(r2_50k)
    N_array = np.arange(1, len(r2_sorted) + 1)
    
    limit = int(0.6 * len(r2_sorted))
    res_rho = linregress(r2_sorted[:limit], N_array[:limit])
    print(f"[Stat] Interior Density (rho) vs r^2: slope = {res_rho.slope:.4f}  (Expected ~ 0.6500)")

    # --- Boundary Dynamics (L vs R) ---
    res_L = linregress(track_R[10:], track_L[10:])
    print(f"[Stat] Boundary Dynamics (L): slope = {res_L.slope:.4f}  (Expected ~ 3.580)")

    # --- Boundary Thickness (\sigma) ---
    log_R = np.log10(track_R[10:])
    log_sigma = np.log10(track_sigma[10:])
    res_sigma = linregress(log_R, log_sigma)
    a = 10 ** res_sigma.intercept
    b = res_sigma.slope
    print(f"[Stat] Boundary Thickness (\sigma): a = {a:.4f}, b = {b:.4f}  (Expected a ~ 0.8813, b ~ 0.396)")

# ==============================================================================
# 3. FIGURE SPECIFICATIONS (MATPLOTLIB)
# ==============================================================================

    out_dir = "09"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating and saving figures to '{out_dir}/' directory...")

    # Figure 1: 3000-cell snapshot (All cells)
    fig, ax = plt.subplots(figsize=(10, 10))
    patches = [Circle((snap_x[i], snap_y[i]), 1.0) for i in range(3000)]
    ax.add_collection(PatchCollection(patches, facecolor='teal', edgecolor='none'))
    ax.set_aspect('equal')
    ax.autoscale_view()
    plt.title("Figure 1: 3,000 Cells Snapshot (All Cells)")
    plt.savefig(os.path.join(out_dir, "Figure_1.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Figure 2: 3000-cell snapshot (True Living Cells)
    fig, ax = plt.subplots(figsize=(10, 10))
    patches = [Circle((snap_x[i], snap_y[i]), 1.0) for i in range(3000) if snap_living[i]]
    ax.add_collection(PatchCollection(patches, facecolor='crimson', edgecolor='none'))
    ax.set_aspect('equal')
    ax.autoscale_view()
    plt.title("Figure 2: 3,000 Cells Snapshot (True Living Cells)")
    plt.savefig(os.path.join(out_dir, "Figure_2.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Figure 3: Enclosed Cells vs r^2 (Full 50k Dataset)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r2_sorted, N_array, color='navy')
    ax.set_xlabel("Radius Squared ($r^2$)")
    ax.set_ylabel("Number of Enclosed Cells ($N$)")
    ax.set_title("Figure 3: Enclosed Cells vs Radius Squared")
    plt.savefig(os.path.join(out_dir, "Figure_3.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Figure 4: Number of Living Cells (L) vs Mean Radius (R)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(track_R, track_L, color='forestgreen')
    ax.set_xlabel("Mean Radius ($R$)")
    ax.set_ylabel("Number of True Living Cells ($L$)")
    ax.set_title("Figure 4: Living Cells vs Mean Radius")
    plt.savefig(os.path.join(out_dir, "Figure_4.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Figure 5: Boundary Thickness (\sigma) vs Mean Radius (R)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(track_R, track_sigma, color='purple')
    ax.set_yscale('log')
    ax.set_xlabel("Mean Radius ($R$)")
    ax.set_ylabel("Boundary Thickness ($\sigma$)")
    ax.set_title("Figure 5: Boundary Thickness vs Mean Radius")
    plt.savefig(os.path.join(out_dir, "Figure_5.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Figure 7: Growth Network / Skeleton of the 3000-cell snapshot
    fig, ax = plt.subplots(figsize=(10, 10))
    lines = []
    for i in range(1, 3000):
        p = snap_parent[i]
        if p != -1:
            lines.append([(snap_x[i], snap_y[i]), (snap_x[p], snap_y[p])])
    ax.add_collection(LineCollection(lines, colors='black', linewidths=0.5))
    ax.set_aspect('equal')
    ax.autoscale_view()
    plt.title("Figure 7: Growth Network Skeleton (3,000 Cells)")
    plt.savefig(os.path.join(out_dir, "Figure_7.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Process Finished Successfully.")
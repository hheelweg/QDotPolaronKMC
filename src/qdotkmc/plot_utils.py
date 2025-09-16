import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# labelling needs to match export_msds
# TODO: need to re-wrtie so that compatible with new export msds format. (09/16/2025)
def plot_msds(filename_csv):

    assert filename_csv.lower().endswith(".csv"), "Expected a .csv file"
    
    # df = pd.read_csv(filename_csv)

    # times = df["time"].values

    # # plot infividual msds
    # for col in df.columns:
    #     if col not in ("time", "ave. msd"):
    #         plt.plot(df["time"], df[col], color = 'C00', alpha = 0.2)

    # # plot mean MSD
    # plt.plot(times, df[ "ave. msd"], label = 'MSD (mean)', color = 'C00')
    # plt.xlabel("time")
    # plt.ylabel("MSD")
    # plt.show()

    df = pd.read_csv(filename_csv)

    # Identify all time and MSD columns by prefix
    time_cols = sorted([col for col in df.columns if col.startswith("time_")])
    msd_cols = sorted([col for col in df.columns if col.startswith("msd_") and col != "mean_msd"])

    if len(time_cols) != len(msd_cols):
        print(f"Warning: Mismatch in time vs msd columns: {len(time_cols)} vs {len(msd_cols)}")

    # Plot individual MSDs
    for t_col, m_col in zip(time_cols, msd_cols):
        if t_col in df.columns and m_col in df.columns:
            plt.plot(df[t_col], df[m_col], alpha=0.25, color='C0')
        else:
            print(f"Skipping: missing {t_col} or {m_col}")

    # Plot mean MSD if available
    if "mean_msd" in df.columns:
        # Construct a "time" axis based on the maximum of all time_i columns
        time_arrays = []
        for t_col in time_cols:
            if t_col in df.columns:
                time_arrays.append(df[t_col])

        if time_arrays:
            time_axis = pd.concat(time_arrays, axis=1).max(axis=1, skipna=True)
            plt.plot(time_axis, df["mean_msd"], label="MSD (mean)", color='C0', linewidth=2.5)

    plt.xlabel("Time")
    plt.ylabel("MSD")
    plt.title("Mean and Individual MSDs")
    plt.legend()
    plt.tight_layout()
    plt.show()


def draw_circle(center, radius):
    # feeds circle-center (np.array), and radius
    return plt.Circle(tuple(center), radius, color = 'grey', alpha = 0.1)


def scatterPoints(points, color, s, label):

    # dimension of qd_lattice
    dim = len(points[0])

    if dim == 2:
        plt.scatter(points.T[0], points.T[1], color = color, s = s, label = label)
    elif dim == 1:
        plt.scatter(points.T[0], np.ones_like(points.T[0]), color = color, s = s, label = label)


# TODO : only implemented for 2D, need to check dimension and adjust accordingly
def plot_lattice(points, qd_lattice, label = 'points', periodic = False):

    # get dimension
    dim = len(qd_lattice[0])

    # if we use periodic boundary conditions for the plottings
    if periodic:
        max_length = np.max(qd_lattice)
        points = np.mod(points, max_length)

    # plot points 
    #plt.scatter(points.T[0], points.T[1], color = 'C01', s = 5, label = label)
    scatterPoints(points, color = 'C01', s = 5, label = label)

    # plot qd_lattice
    #plt.scatter(qd_lattice.T[0], qd_lattice.T[1], color = 'k', s = 2, label = 'QD lattice')
    scatterPoints(qd_lattice, color = 'k', s = 2, label = 'QD lattice')
    if dim == 2:
        plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

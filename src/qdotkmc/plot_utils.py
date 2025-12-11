import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# labelling needs to match export_msds
def plot_msds(filename_csv):

    assert filename_csv.lower().endswith(".csv"), "Expected a .csv file"
    
    df = pd.read_csv(filename_csv)

    # identify all time and MSD columns by prefix
    time_cols = sorted([col for col in df.columns if col.startswith("time_")])
    msd_cols = sorted([col for col in df.columns if col.startswith("msd_") and col != "mean_msd"])

    # plot individual MSDs
    for t_col, m_col in zip(time_cols, msd_cols):
        if t_col in df.columns and m_col in df.columns:
            plt.plot(df[t_col], df[m_col], alpha=0.25, color='C0')
        else:
            print(f"Skipping: missing {t_col} or {m_col}")

    # plot mean MSD if available
    if "mean_msd" in df.columns:
        # construct a "time" axis based on the maximum of all time_i columns
        time_arrays = []
        for t_col in time_cols:
            if t_col in df.columns:
                time_arrays.append(df[t_col])

        if time_arrays:
            time_axis = pd.concat(time_arrays, axis=1).max(axis=1, skipna=True)
            plt.plot(time_axis, df["mean_msd"], label="MSD (mean)", color='C0', linewidth=2.5)

    # TODO : add units
    plt.xlabel("time")
    plt.ylabel("msd")
    plt.legend()
    plt.tight_layout()
    plt.show()


def draw_circle(center, radius):
    # feeds circle-center (np.array), and radius
    return plt.Circle(tuple(center), radius, color = 'grey', alpha = 0.1)


def scatterPoints(points, color, s, label):

    points = np.asarray(points)
    dim = points.shape[1]

    if ax is None:
        # create suitable axes if not provided
        if dim == 3:
            ax = plt.axes(projection="3d")
        else:
            ax = plt.gca()

    if dim == 2:
        ax.scatter(points[:, 0], points[:, 1], color=color, s=s, label=label)
    elif dim == 1:
        # arbitrary y = 0 for 1D visualization
        ax.scatter(points[:, 0], np.zeros_like(points[:, 0]),
                   color=color, s=s, label=label)
    elif dim == 3:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   color=color, s=s, label=label)
    else:
        raise ValueError(f"scatterPoints: dim={dim} not supported (must be 1,2,3)")

    return ax


def plot_lattice(points, qd_lattice, label='points', periodic=False, ax=None):
    """
    Plot QD lattice + points in 1D/2D/3D.
    points, qd_lattice: (N, dim)
    periodic: if True, wrap points into box defined by qd_lattice extents.
    """
    points = np.asarray(points)
    qd_lattice = np.asarray(qd_lattice)

    dim = qd_lattice.shape[1]

    # create appropriate axes if not given
    if ax is None:
        if dim == 3:
            ax = plt.axes(projection="3d")
        else:
            ax = plt.gca()

    if periodic:
        # infer box lengths per dimension from lattice extents
        L = np.max(qd_lattice, axis=0)                  # assumes coords in [0, L_d)
        L[L == 0] = 1.0                                 # avoid division by zero if degenerate dim
        points = np.mod(points, L)

    # plot points
    ax = scatterPoints(points, color='C1', s=5, label=label, ax=ax)

    # plot QD lattice
    ax = scatterPoints(qd_lattice, color='k', s=2, label='QD lattice', ax=ax)

    if dim == 2:
        ax.set_aspect('equal', adjustable='box')
    elif dim == 3:
        # make the box roughly cubic for nicer plots
        xyz_min = qd_lattice.min(axis=0)
        xyz_max = qd_lattice.max(axis=0)
        ranges = xyz_max - xyz_min
        max_range = ranges.max()
        mid = (xyz_max + xyz_min) / 2.0
        ax.set_box_aspect(max_range * np.ones(3))
        ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
        ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
        ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

    ax.axis('off')
    return ax

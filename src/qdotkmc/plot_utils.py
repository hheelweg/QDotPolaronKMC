import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# labelling needs to match export_msds
def plot_msds(filename_csv):

    assert filename_csv.lower().endswith(".csv"), "Expected a .csv file"
    
    df = pd.read_csv("msds.csv")

    times = df["time"].values

    # plot infividual msds
    for col in df.columns:
        if col not in ("time", "ave. msd"):
            plt.plot(df["time"], df[col], color = 'C00', alpha = 0.2)

    # plot mean MSD
    plt.plot(times, df[ "ave. msd"], label = 'MSD (mean)', color = 'C00')
    plt.xlabel("time")
    plt.ylabel("MSD")
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

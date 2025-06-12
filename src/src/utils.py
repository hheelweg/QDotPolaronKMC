import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

"""
This file contains some helper functions that will be called
"""


def to_liouville(rho):
    if len(rho.shape) == 2:
        return rho.flatten().astype(np.complex_)
    else:
        # A tensor to a matrix 
        ns = rho.shape[0]
        rho_mat = np.zeros((ns*ns,ns*ns), dtype=np.complex_) 
        I = 0
        for i in range(ns):
            for j in range(ns):
                J = 0
                for k in range(ns):
                    for l in range(ns):
                        rho_mat[I,J] = rho[i,j,k,l]
                        J += 1
                I += 1
        return rho_mat


def diagonalize(H,S=None):
    """
    Diagonalize a real, symmetrix matrix and return sorted results.
    
    Return the eigenvalues and eigenvectors (column matrix) 
    sorted from lowest to highest eigenvalue.
    """
    E,C = la.eigh(H,S)
    E = np.real(E)
    #C = np.real(C)

    idx = E.argsort()
    #idx = (-E).argsort()
    E = E[idx]
    C = C[:,idx]

    return E,C


def transform_rho(transform, rhos):
    rhos = np.array(rhos)
    rhos_trans = list()
    if rhos.ndim == 3:
        for rho in rhos:
            rhos_trans.append(transform(rho))
    else:
        rhos_trans = transform(rhos)
    return np.array(rhos_trans)


def matrix_dot(*matrices):
    """Calculate the matrix product of multiple matrices."""
    A = matrices[0].copy()
    for B in matrices[1:]:
        A = np.dot(A,B)
    return A

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



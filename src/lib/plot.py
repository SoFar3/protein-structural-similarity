import sys, os; import sys, os; sys.path.append(os.path.abspath(os.path.join(os.curdir, "src")))

import lib.analysis as als

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook, cm
from matplotlib.colors import LightSource

def simple_plot(x, title="Simple plot"):
    plt.plot(np.arange(0, x.shape[0]), x)
    plt.title(title)
    plt.show()

def graph3d(data, title="3D Graph"):
    print(data.shape)
    X, Y = np.meshgrid(np.arange(0, data.shape[1]), np.arange(0, data.shape[0]))

    plt.clf()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ls = LightSource(270, 45)
    rgb = ls.shade(data, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')

    ax.plot_surface(X, Y, data, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=1, antialiased=False, shade=True)

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()
    
def heatmap2d(data, scale=1):
    fig, ax1 = plt.subplots(figsize=(10 * scale, 5 * scale))
    pos = ax1.imshow(data, cmap='hot')
    fig.colorbar(pos, ax=ax1, shrink=0.7)
    plt.show()

def scatter2d(data):
    graph = plt.scatter(data[:, 0], data[:, 1], c=np.arange(0, data.shape[0]))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(graph)
    plt.show()

def scatter3d(data):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(projection = '3d')
    c = np.arange(len(data)) / len(data)
    p = ax.scatter(
        xs=data[:, 0], 
        ys=data[:, 1], 
        zs=data[:, 2], 
        c=np.arange(0, data.shape[0])    
    )
    fig.colorbar(p, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
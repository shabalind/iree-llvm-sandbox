import matplotlib.pyplot as plt
import numpy as np

def show_plot_2d(xs, ys):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(xs, ys, c='r', marker='o')
  ax.set_xlabel('ops')
  ax.set_ylabel('dims')
  plt.savefig("fig2d.png")

def show_plot_3d(xs, ys, zs):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(xs, ys, zs, c='r', marker='o')
  ax.set_xlabel('ops')
  ax.set_ylabel('vars')
  ax.set_zlabel('dims')
  plt.savefig("fig3d.png")

if __name__ == "__main__":
  points = [
(5, 3, 3), (5, 4, 2), (0, 1, 0), (2, 2, 2), (6, 3, 2), (4, 2, 1), (4, 4, 4), (3, 2, 1), (1, 2, 2), (5, 2, 1), (6, 5, 2), (5, 5, 3), (3, 4, 4), (3, 3, 2), (4, 3, 2), (2, 3, 3), (6, 4, 3), (5, 3, 2), (5, 4, 1), (5, 4, 4), (6, 1, 1), (2, 2, 1), (0, 1, 2), (6, 3, 1), (6, 5, 4), (3, 1, 1), (4, 4, 3), (1, 2, 1), (3, 4, 3), (3, 3, 1), (6, 2, 2), (4, 3, 1), (2, 3, 2), (4, 1, 1), (6, 4, 2), (5, 1, 1), (5, 4, 3), (5, 3, 1), (0, 1, 1), (4, 2, 2), (2, 1, 1), (4, 5, 4), (4, 4, 2), (5, 2, 2), (6, 3, 3), (6, 5, 3), (3, 2, 2), (1, 2, 3), (3, 3, 3), (5, 5, 4), (2, 3, 4), (6, 2, 1), (6, 4, 4), (4, 3, 3), (1, 1, 1), (6, 4, 1)
]
  nops = [x for (x, _, _) in points]
  nvars = [x for (_, x, _) in points]
  ndims = [x for (_, _, x) in points]
  show_plot_2d(nops, ndims)


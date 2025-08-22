import numpy as np
from numpy import trapz


def make_tangent(d, n):
  t = np.cross(d, n)
  t /= np.linalg.norm(t, axis=-1, keepdims=True)
  return np.cross(t, d)


def unit_circle(n):
  a = np.linspace(0, 2 * np.pi, n + 1)[:-1]
  return np.stack( [np.sin(a), np.cos(a)], axis=1)


def vertex_dirs(points):
  d = points[1:] - points[:-1]
  d = d / np.linalg.norm(d)
  
  smooth = (d[1:] + d[:-1]) * 0.5
  dirs = np.concatenate([
    np.array(d[0:1]), smooth, np.array(d[-2:-1])
  ])

  return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def gen_tangents(dirs, t):
  tangents = []

  for dir in dirs:
    t = make_tangent(dir, t)
    tangents.append(t)

  return np.stack(tangents)


def random_unit(dtype=np.float32):
  x =  np.random.randn(3).astype(dtype)
  return x / np.linalg.norm(x)


def calculate_AuC(y, dx=0.01):
  return trapz(y=y, dx=dx)
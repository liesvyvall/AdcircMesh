"""Operaciones de edicion manual, todas reversibles.

Cada operacion se graba en un `MeshOp` que sabe deshacerse y rehacerse.  Como
la malla nunca borra fisicamente nodos ni elementos (solo los marca muertos),
los indices son estables y el registro es un simple delta.
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh


class MeshOp:
    """Delta reversible de una edicion."""

    def __init__(self, mesh: Mesh, name: str):
        self.mesh = mesh
        self.name = name
        self.moved = []          # (i, old_xy, new_xy)
        self.depths = []         # (i, old_z, new_z)
        self.added_nodes = []    # indices
        self.added_elems = []    # indices
        self.changed_elems = []  # (k, old_row, new_row)
        self.killed_elems = []
        self.killed_nodes = []
        self.bnd_before = None
        self.bnd_after = None

    def __bool__(self):
        return bool(self.moved or self.depths or self.added_nodes or
                    self.added_elems or self.changed_elems or
                    self.killed_elems or self.killed_nodes or
                    self.bnd_before is not None)

    # ------------------------------------------------------------------ undo
    def undo(self):
        m = self.mesh
        for i, old, _ in self.moved:
            m.xy[i] = old
        for i, old, _ in self.depths:
            m.z[i] = old
        for k, old, _ in self.changed_elems:
            m.tri[k] = old
        for k in self.added_elems:
            m.elem_alive[k] = False
        for i in self.added_nodes:
            m.node_alive[i] = False
        for k in self.killed_elems:
            m.elem_alive[k] = True
        for i in self.killed_nodes:
            m.node_alive[i] = True
        if self.bnd_before is not None:
            m.boundaries = [b.copy() for b in self.bnd_before]
        m.touch_topology()

    def redo(self):
        m = self.mesh
        for i, _, new in self.moved:
            m.xy[i] = new
        for i, _, new in self.depths:
            m.z[i] = new
        for k, _, new in self.changed_elems:
            m.tri[k] = new
        for k in self.added_elems:
            m.elem_alive[k] = True
        for i in self.added_nodes:
            m.node_alive[i] = True
        for k in self.killed_elems:
            m.elem_alive[k] = False
        for i in self.killed_nodes:
            m.node_alive[i] = False
        if self.bnd_after is not None:
            m.boundaries = [b.copy() for b in self.bnd_after]
        m.touch_topology()


class SnapshotOp:
    """Operacion global reversible por instantanea completa (reparaciones)."""

    def __init__(self, mesh: Mesh, name: str, before, after):
        self.mesh, self.name = mesh, name
        self._before, self._after = before, after

    def __bool__(self):
        return True

    def undo(self):
        self.mesh.restore(self._before)

    def redo(self):
        self.mesh.restore(self._after)


class Recorder:
    """Aplica cambios sobre la malla mientras los graba en un `MeshOp`."""

    def __init__(self, mesh: Mesh, name: str):
        self.m = mesh
        self.op = MeshOp(mesh, name)
        self._bnd_captured = False

    # ------------------------------------------------------------ primitivas
    def move_node(self, i, x, y):
        i = int(i)
        old = self.m.xy[i].copy()
        self.m.xy[i] = (x, y)
        self.op.moved.append((i, old, np.array([x, y], float)))

    def set_depth(self, i, z):
        i = int(i)
        old = float(self.m.z[i])
        self.m.z[i] = z
        self.op.depths.append((i, old, float(z)))

    def add_node(self, x, y, z):
        self.m.xy = np.vstack([self.m.xy, [x, y]])
        self.m.z = np.append(self.m.z, z)
        self.m.node_alive = np.append(self.m.node_alive, True)
        i = len(self.m.xy) - 1
        self.op.added_nodes.append(i)
        return i

    def add_element(self, a, b, c):
        pm = self.m.pm if len(self.m.xy) == len(self.m.pm) else self.m.to_meters()
        row = _ccw(pm, int(a), int(b), int(c))
        self.m.tri = np.vstack([self.m.tri, row])
        self.m.elem_alive = np.append(self.m.elem_alive, True)
        k = len(self.m.tri) - 1
        self.op.added_elems.append(k)
        return k

    def set_element(self, k, a, b, c):
        k = int(k)
        old = self.m.tri[k].copy()
        new = np.array([int(a), int(b), int(c)], np.int64)
        self.m.tri[k] = new
        self.op.changed_elems.append((k, old, new))

    def kill_element(self, k):
        k = int(k)
        if self.m.elem_alive[k]:
            self.m.elem_alive[k] = False
            self.op.killed_elems.append(k)

    def kill_node(self, i):
        i = int(i)
        if self.m.node_alive[i]:
            self.m.node_alive[i] = False
            self.op.killed_nodes.append(i)

    def capture_boundaries(self):
        if not self._bnd_captured:
            self.op.bnd_before = [b.copy() for b in self.m.boundaries]
            self._bnd_captured = True

    def set_boundaries(self, bnds):
        self.capture_boundaries()
        self.m.boundaries = list(bnds)
        self.op.bnd_after = [b.copy() for b in bnds]

    def finish(self):
        self.m.touch_topology()
        return self.op


def _ccw(pm, a, b, c):
    """Devuelve el triangulo orientado en sentido antihorario."""
    A, B, C = pm[a], pm[b], pm[c]
    cross = (B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])
    return np.array([a, b, c] if cross >= 0 else [a, c, b], np.int64)


# ---------------------------------------------------------------- operaciones
def move_nodes(mesh, ids, new_xy, name="Mover nodos"):
    r = Recorder(mesh, name)
    for i, p in zip(np.atleast_1d(ids), np.atleast_2d(new_xy)):
        r.move_node(i, p[0], p[1])
    return r.finish()


def set_depths(mesh, ids, values, name="Cambiar profundidad"):
    r = Recorder(mesh, name)
    vals = np.broadcast_to(np.atleast_1d(values), np.atleast_1d(ids).shape)
    for i, v in zip(np.atleast_1d(ids), vals):
        r.set_depth(i, v)
    return r.finish()


def delete_elements(mesh, ids, name="Borrar elementos"):
    r = Recorder(mesh, name)
    for k in np.atleast_1d(ids):
        r.kill_element(k)
    return r.finish()


def delete_nodes(mesh, ids, name="Borrar nodos"):
    """Borra nodos y todos los elementos que los usan."""
    r = Recorder(mesh, name)
    for i in np.atleast_1d(ids):
        for k in mesh.elems_of_node(int(i)):
            r.kill_element(k)
        r.kill_node(i)
    return r.finish()


def add_node(mesh, x, y, depth=None, name="Anadir nodo"):
    """Anade un nodo; si cae dentro de un elemento, lo subdivide en 3."""
    k = mesh.pick_element(x, y)
    r = Recorder(mesh, name)
    if k is None:
        r.add_node(x, y, 0.0 if depth is None else depth)
        return r.finish()
    tri = mesh.tri[k]
    d = _interp_depth(mesh, tri, x, y) if depth is None else depth
    n = r.add_node(x, y, d)
    r.kill_element(k)
    a, b, c = (int(v) for v in tri)
    for u, v in ((a, b), (b, c), (c, a)):
        r.add_element(u, v, n)
    return r.finish()


def _interp_depth(mesh, tri, x, y):
    """Profundidad por coordenadas baricentricas dentro del triangulo."""
    p = mesh.to_meters(np.array([[x, y]]))[0]
    A, B, C = mesh.pm[tri]
    d = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    if abs(d) < 1e-12:
        return float(mesh.z[tri].mean())
    l1 = ((B[1] - C[1]) * (p[0] - C[0]) + (C[0] - B[0]) * (p[1] - C[1])) / d
    l2 = ((C[1] - A[1]) * (p[0] - C[0]) + (A[0] - C[0]) * (p[1] - C[1])) / d
    l3 = 1.0 - l1 - l2
    return float(l1 * mesh.z[tri[0]] + l2 * mesh.z[tri[1]] + l3 * mesh.z[tri[2]])


def split_edge(mesh, a, b, name="Dividir arista"):
    """Inserta un nodo en el punto medio y parte los 1-2 elementos incidentes."""
    a, b = int(a), int(b)
    els = [int(k) for k in mesh.elems_of_node(a) if b in mesh.tri[k]]
    if not els:
        return None
    r = Recorder(mesh, name)
    mid = mesh.xy[[a, b]].mean(axis=0)
    n = r.add_node(mid[0], mid[1], float(mesh.z[[a, b]].mean()))
    for k in els:
        c = int([v for v in mesh.tri[k] if v not in (a, b)][0])
        r.kill_element(k)
        r.add_element(a, n, c)
        r.add_element(n, b, c)
    return r.finish()


def swap_edge(mesh, a, b, name="Voltear arista"):
    """Voltea la diagonal del cuadrilatero formado por los 2 elementos."""
    a, b = int(a), int(b)
    els = [int(k) for k in mesh.elems_of_node(a) if b in mesh.tri[k]]
    if len(els) != 2:
        return None
    k1, k2 = els
    c = int([v for v in mesh.tri[k1] if v not in (a, b)][0])
    d = int([v for v in mesh.tri[k2] if v not in (a, b)][0])
    if c == d:
        return None
    pm = mesh.pm
    if _area(pm, c, d, a) <= 0 or _area(pm, d, c, b) <= 0:
        return None                       # el cuadrilatero no es convexo
    r = Recorder(mesh, name)
    r.set_element(k1, c, d, a)
    r.set_element(k2, d, c, b)
    return r.finish()


def _area(pm, a, b, c):
    A, B, C = pm[a], pm[b], pm[c]
    return 0.5 * ((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1]))


def split_element(mesh, k, name="Dividir elemento"):
    """Inserta el centroide y parte el elemento en 3."""
    k = int(k)
    tri = mesh.tri[k]
    cxy = mesh.xy[tri].mean(axis=0)
    r = Recorder(mesh, name)
    n = r.add_node(cxy[0], cxy[1], float(mesh.z[tri].mean()))
    r.kill_element(k)
    a, b, c = (int(v) for v in tri)
    for u, v in ((a, b), (b, c), (c, a)):
        r.add_element(u, v, n)
    return r.finish()


def create_element(mesh, a, b, c, name="Crear elemento"):
    a, b, c = int(a), int(b), int(c)
    if len({a, b, c}) != 3:
        return None
    existing = {tuple(sorted(mesh.tri[k])) for k in mesh.elems_of_node(a)}
    if tuple(sorted((a, b, c))) in existing:
        return None
    r = Recorder(mesh, name)
    r.add_element(a, b, c)
    return r.finish()


def merge_nodes(mesh, keep, drop, name="Fusionar nodos"):
    """Colapsa `drop` sobre `keep`, eliminando los elementos que degeneran."""
    keep, drop = int(keep), int(drop)
    if keep == drop:
        return None
    r = Recorder(mesh, name)
    for k in list(mesh.elems_of_node(drop)):
        k = int(k)
        row = [keep if v == drop else int(v) for v in mesh.tri[k]]
        if len(set(row)) < 3:
            r.kill_element(k)
        else:
            r.set_element(k, *row)
    r.kill_node(drop)
    return r.finish()


def collapse_edge(mesh, a, b, name="Colapsar arista"):
    """Fusiona los dos extremos en su punto medio."""
    a, b = int(a), int(b)
    mid = mesh.xy[[a, b]].mean(axis=0)
    zmid = float(mesh.z[[a, b]].mean())
    op = merge_nodes(mesh, a, b, name)
    if op is None:
        return None
    r = Recorder(mesh, name)
    r.move_node(a, mid[0], mid[1])
    r.set_depth(a, zmid)
    sub = r.finish()
    op.moved += sub.moved
    op.depths += sub.depths
    return op


def smooth_nodes(mesh, ids, iters=20, keep_boundary=True, relax=0.6,
                 name="Suavizar"):
    """Laplaciano local con garantia de no invertir elementos."""
    ids = np.atleast_1d(np.asarray(ids, np.int64))
    bmask = mesh.boundary_node_mask()
    if keep_boundary:
        ids = ids[~bmask[ids]]
    ids = ids[mesh.node_alive[ids]]
    if len(ids) == 0:
        return None
    pm = mesh.pm.copy()
    orig = pm.copy()
    for _ in range(iters):
        for i in ids:
            nb = mesh.neighbors_of_node(int(i))
            if len(nb) < 3:
                continue
            els = mesh.elems_of_node(int(i))
            q0 = _min_quality(pm, mesh.tri[els])
            old = pm[i].copy()
            step = pm[nb].mean(axis=0) - old
            best, bq = None, q0
            for frac in (relax, 0.5 * relax, 0.25 * relax):
                pm[i] = old + frac * step
                q = _min_quality(pm, mesh.tri[els])
                if q > bq + 1e-12:
                    best, bq = pm[i].copy(), q
            pm[i] = best if best is not None else old
    moved = np.flatnonzero(np.linalg.norm(pm - orig, axis=1) > 1e-9)
    if len(moved) == 0:
        return None
    new_xy = mesh.to_degrees(pm[moved])
    return move_nodes(mesh, moved, new_xy, name)


def _min_quality(pm, tt):
    a, b, c = pm[tt[:, 0]], pm[tt[:, 1]], pm[tt[:, 2]]
    ar = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    l2 = ((b - a) ** 2).sum(1) + ((c - b) ** 2).sum(1) + ((a - c) ** 2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 4 * np.sqrt(3) * ar / np.where(l2 > 0, l2, np.nan)   # firmado
    return float(np.nan_to_num(q, nan=-1.0).min())


def rings_around(mesh, ids, rings=2):
    """Vecindad de N anillos alrededor de un conjunto de nodos."""
    sel = np.zeros(len(mesh.xy), bool)
    sel[np.atleast_1d(ids)] = True
    indptr, data = mesh.node_adj()
    for _ in range(rings):
        cur = np.flatnonzero(sel)
        for i in cur:
            sel[data[indptr[i]:indptr[i + 1]]] = True
    return np.flatnonzero(sel)


def nodes_of_elements(mesh, elem_ids):
    ids = np.atleast_1d(np.asarray(elem_ids, np.int64))
    return np.unique(mesh.tri[ids].ravel()) if len(ids) else np.zeros(0, np.int64)

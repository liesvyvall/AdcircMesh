"""Nucleo de datos: malla triangular no estructurada para ADCIRC/SWAN.

Diseno clave: los nodos y elementos NUNCA se borran fisicamente durante la
edicion, solo se marcan como muertos (`node_alive` / `elem_alive`).  Asi los
indices son estables y el undo/redo es trivial.  La compactacion ocurre al
guardar.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial import cKDTree

R_EARTH = 6378206.4          # m, esferoide usado por ADCIRC
G = 9.80665

OCEAN, LAND, ISLAND = "ocean", "land", "island"

# IBTYPE tipicos de ADCIRC para fronteras de tierra
IBTYPE_LABELS = {
    0:  "0  - tierra natural, sin flujo normal",
    1:  "1  - isla, sin flujo normal",
    2:  "2  - tierra con flujo entrante especificado",
    10: "10 - tierra natural, deslizamiento libre",
    11: "11 - isla, deslizamiento libre",
    12: "12 - tierra con flujo, deslizamiento libre",
    20: "20 - tierra natural, sin flujo normal (barrera)",
    21: "21 - isla, sin flujo normal (barrera)",
    22: "22 - tierra con flujo (barrera)",
    30: "30 - frontera radiante externa",
}


class Boundary:
    """Un nodestring de frontera, al estilo de SMS."""

    __slots__ = ("kind", "ibtype", "nodes", "name", "extra")

    def __init__(self, kind, nodes, ibtype=None, name="", extra=None):
        self.kind = kind                       # OCEAN | LAND | ISLAND
        self.nodes = np.asarray(nodes, np.int64).ravel()
        self.ibtype = ibtype
        self.name = name or ""
        self.extra = extra                     # tokens extra (barreras/weirs)

    def __len__(self):
        return len(self.nodes)

    def copy(self):
        return Boundary(self.kind, self.nodes.copy(), self.ibtype, self.name,
                        list(self.extra) if self.extra else None)

    @property
    def label(self):
        if self.kind == OCEAN:
            return f"Oceano ({len(self.nodes)} nodos)"
        t = "Isla" if self.kind == ISLAND else "Tierra"
        return f"{t} IBTYPE={self.ibtype} ({len(self.nodes)} nodos)"


class Mesh:
    """Malla triangular con proyeccion CPP local y caches invalidables."""

    def __init__(self, xy, z, tri, boundaries=None, title="mesh"):
        self.xy = np.asarray(xy, float).reshape(-1, 2).copy()
        self.z = np.asarray(z, float).ravel().copy()
        self.tri = np.asarray(tri, np.int64).reshape(-1, 3).copy()
        self.node_alive = np.ones(len(self.xy), bool)
        self.elem_alive = np.ones(len(self.tri), bool)
        self.boundaries = list(boundaries or [])
        self.title = title
        self.lon0 = math.radians(float(self.xy[:, 0].mean()))
        self.lat0 = math.radians(float(self.xy[:, 1].mean()))
        self._gcache = {}          # depende de coordenadas
        self._tcache = {}          # depende de conectividad
        self.geom_version = 0
        self.topo_version = 0

    # ------------------------------------------------------------ versiones
    def touch_geometry(self):
        self._gcache.clear()
        self.geom_version += 1

    def touch_topology(self):
        self._gcache.clear()
        self._tcache.clear()
        self.topo_version += 1
        self.geom_version += 1

    # ---------------------------------------------------------- proyeccion
    def to_meters(self, xy=None):
        xy = self.xy if xy is None else np.atleast_2d(xy)
        return np.column_stack([
            R_EARTH * (np.deg2rad(xy[:, 0]) - self.lon0) * math.cos(self.lat0),
            R_EARTH * np.deg2rad(xy[:, 1]),
        ])

    def to_degrees(self, pm):
        pm = np.atleast_2d(pm)
        return np.column_stack([
            np.degrees(pm[:, 0] / (R_EARTH * math.cos(self.lat0)) + self.lon0),
            np.degrees(pm[:, 1] / R_EARTH),
        ])

    @property
    def pm(self):
        """Coordenadas proyectadas en metros (cache)."""
        v = self._gcache.get("pm")
        if v is None:
            v = self._gcache["pm"] = self.to_meters()
        return v

    # ------------------------------------------------------------- conteos
    @property
    def n_nodes(self):
        return int(self.node_alive.sum())

    @property
    def n_elems(self):
        return int(self.elem_alive.sum())

    def bbox(self):
        m = self.node_alive
        if not m.any():
            return (0.0, 0.0, 1.0, 1.0)
        p = self.xy[m]
        return (float(p[:, 0].min()), float(p[:, 1].min()),
                float(p[:, 0].max()), float(p[:, 1].max()))

    # ------------------------------------------------------ subconjunto vivo
    def live_elems(self):
        """(ids_globales, tri_vivos)"""
        v = self._tcache.get("live_elems")
        if v is None:
            ids = np.flatnonzero(self.elem_alive)
            v = self._tcache["live_elems"] = (ids, self.tri[ids])
        return v

    def live_nodes(self):
        v = self._tcache.get("live_nodes")
        if v is None:
            v = self._tcache["live_nodes"] = np.flatnonzero(self.node_alive)
        return v

    # -------------------------------------------------------------- aristas
    def edges(self):
        """(ue, ce): aristas unicas ordenadas y su multiplicidad."""
        v = self._tcache.get("edges")
        if v is None:
            _, t = self.live_elems()
            if len(t) == 0:
                v = (np.zeros((0, 2), np.int64), np.zeros(0, np.int64))
            else:
                e = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
                e = np.sort(e, axis=1)
                ue, ce = np.unique(e, axis=0, return_counts=True)
                v = (ue, ce)
            self._tcache["edges"] = v
        return v

    def boundary_edges(self):
        ue, ce = self.edges()
        return ue[ce == 1]

    def boundary_node_mask(self):
        v = self._tcache.get("bmask")
        if v is None:
            v = np.zeros(len(self.xy), bool)
            be = self.boundary_edges()
            if len(be):
                v[be.ravel()] = True
            self._tcache["bmask"] = v
        return v

    # ---------------------------------------------------- adyacencias (CSR)
    def node_elems(self):
        """CSR nodo -> elementos incidentes vivos: (indptr, data)."""
        v = self._tcache.get("node_elems")
        if v is None:
            ids, t = self.live_elems()
            nn = len(self.xy)
            if len(t) == 0:
                v = (np.zeros(nn + 1, np.int64), np.zeros(0, np.int64))
            else:
                nodes = t.ravel()
                elems = np.repeat(ids, 3)
                order = np.argsort(nodes, kind="stable")
                data = elems[order]
                counts = np.bincount(nodes, minlength=nn)
                indptr = np.zeros(nn + 1, np.int64)
                np.cumsum(counts, out=indptr[1:])
                v = (indptr, data)
            self._tcache["node_elems"] = v
        return v

    def elems_of_node(self, i):
        indptr, data = self.node_elems()
        return data[indptr[i]:indptr[i + 1]]

    def node_adj(self):
        """CSR nodo -> nodos vecinos: (indptr, data)."""
        v = self._tcache.get("node_adj")
        if v is None:
            ue, _ = self.edges()
            nn = len(self.xy)
            if len(ue) == 0:
                v = (np.zeros(nn + 1, np.int64), np.zeros(0, np.int64))
            else:
                a = np.concatenate([ue[:, 0], ue[:, 1]])
                b = np.concatenate([ue[:, 1], ue[:, 0]])
                order = np.argsort(a, kind="stable")
                data = b[order]
                counts = np.bincount(a, minlength=nn)
                indptr = np.zeros(nn + 1, np.int64)
                np.cumsum(counts, out=indptr[1:])
                v = (indptr, data)
            self._tcache["node_adj"] = v
        return v

    def neighbors_of_node(self, i):
        indptr, data = self.node_adj()
        return data[indptr[i]:indptr[i + 1]]

    def valence(self):
        indptr, _ = self.node_adj()
        return np.diff(indptr)

    # -------------------------------------------------------------- picking
    def node_tree(self):
        v = self._gcache.get("tree")
        if v is None:
            live = self.live_nodes()
            if len(live) == 0:
                return None, live
            v = self._gcache["tree"] = (cKDTree(self.pm[live]), live)
        return v

    def pick_node(self, x, y, radius_m=None):
        """Nodo vivo mas cercano a (lon, lat). Devuelve indice global o None."""
        res = self.node_tree()
        if res is None or res[0] is None:
            return None
        tree, live = res
        p = self.to_meters(np.array([[x, y]]))[0]
        d, k = tree.query(p)
        if radius_m is not None and d > radius_m:
            return None
        return int(live[k])

    def pick_element(self, x, y, search=48):
        """Elemento vivo que contiene (lon, lat), o None."""
        res = self.node_tree()
        if res is None or res[0] is None:
            return None
        tree, live = res
        p = self.to_meters(np.array([[x, y]]))[0]
        k = min(search, len(live))
        _, idx = tree.query(p, k=k)
        idx = np.atleast_1d(idx)
        seen = set()
        pm = self.pm
        for n in live[idx]:
            for e in self.elems_of_node(int(n)):
                e = int(e)
                if e in seen:
                    continue
                seen.add(e)
                a, b, c = pm[self.tri[e]]
                d1 = (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])
                d2 = (c[0] - b[0]) * (p[1] - b[1]) - (p[0] - b[0]) * (c[1] - b[1])
                d3 = (a[0] - c[0]) * (p[1] - c[1]) - (p[0] - c[0]) * (a[1] - c[1])
                if (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0):
                    return e
        return None

    def pick_edge(self, x, y):
        """Arista viva mas cercana a (lon, lat): (a, b) o None."""
        n = self.pick_node(x, y)
        if n is None:
            return None
        pm = self.pm
        p = self.to_meters(np.array([[x, y]]))[0]
        best, bd = None, np.inf
        for m in self.neighbors_of_node(n):
            a, b = pm[n], pm[m]
            ab = b - a
            L2 = float(ab @ ab)
            tpar = 0.0 if L2 == 0 else max(0.0, min(1.0, float((p - a) @ ab) / L2))
            d = float(np.linalg.norm(p - (a + tpar * ab)))
            if d < bd:
                bd, best = d, (int(n), int(m))
        return best

    # ----------------------------------------------------- edicion primitiva
    def add_node(self, x, y, depth):
        self.xy = np.vstack([self.xy, [x, y]])
        self.z = np.append(self.z, depth)
        self.node_alive = np.append(self.node_alive, True)
        self.touch_topology()
        return len(self.xy) - 1

    def add_element(self, a, b, c):
        self.tri = np.vstack([self.tri, [a, b, c]])
        self.elem_alive = np.append(self.elem_alive, True)
        self.touch_topology()
        return len(self.tri) - 1

    def set_element(self, k, a, b, c):
        self.tri[k] = (a, b, c)
        self.touch_topology()

    def kill_element(self, k):
        self.elem_alive[k] = False
        self.touch_topology()

    def revive_element(self, k):
        self.elem_alive[k] = True
        self.touch_topology()

    def kill_node(self, i):
        self.node_alive[i] = False
        self.touch_topology()

    def revive_node(self, i):
        self.node_alive[i] = True
        self.touch_topology()

    def move_node(self, i, x, y):
        self.xy[i] = (x, y)
        self.touch_geometry()

    # ---------------------------------------------------------- compactacion
    def compacted(self):
        """(xy, z, tri, boundaries) con indices densos 0-based, listo para E/S.

        Los nodos huerfanos (vivos pero sin elemento) se descartan.
        """
        ids, t = self.live_elems()
        nn = len(self.xy)
        used = np.zeros(nn, bool)
        if len(t):
            used[t.ravel()] = True
        used &= self.node_alive
        remap = -np.ones(nn, np.int64)
        remap[used] = np.arange(used.sum())
        newb = []
        for b in self.boundaries:
            keep = remap[b.nodes] >= 0
            nb = remap[b.nodes][keep]
            ex = [e for e, k in zip(b.extra, keep) if k] if b.extra else None
            if len(nb) >= 2:
                newb.append(Boundary(b.kind, nb, b.ibtype, b.name, ex))
        return self.xy[used], self.z[used], remap[t], newb

    def snapshot(self):
        return dict(
            xy=self.xy.copy(), z=self.z.copy(), tri=self.tri.copy(),
            node_alive=self.node_alive.copy(), elem_alive=self.elem_alive.copy(),
            boundaries=[b.copy() for b in self.boundaries],
        )

    def restore(self, snap):
        self.xy = snap["xy"].copy()
        self.z = snap["z"].copy()
        self.tri = snap["tri"].copy()
        self.node_alive = snap["node_alive"].copy()
        self.elem_alive = snap["elem_alive"].copy()
        self.boundaries = [b.copy() for b in snap["boundaries"]]
        self.touch_topology()

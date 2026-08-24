"""Herramientas de reparacion global de la malla.

Todas mutan la malla in situ y devuelven un texto con lo que hicieron.  El
undo se maneja por instantanea (`SnapshotOp`) desde la interfaz.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, reverse_cuthill_mckee
from scipy.spatial import cKDTree

from .mesh import Mesh, Boundary, OCEAN, LAND, ISLAND
from .quality import signed_area, edge_lengths, shape_quality, angles_deg


# ------------------------------------------------------------------ auxiliar
def _live(mesh):
    return mesh.live_elems()


def boundary_loops(mesh):
    """Lazos cerrados de la frontera, el mayor primero."""
    be = mesh.boundary_edges()
    adj = defaultdict(list)
    for a, b in be:
        adj[a].append(b); adj[b].append(a)
    seen, loops = set(), []
    for s in adj:
        if s in seen:
            continue
        loop = [s]; seen.add(s); cur, prev = s, None
        while True:
            nxt = [k for k in adj[cur] if k != prev and k not in seen]
            if not nxt:
                break
            k = nxt[0]; loop.append(k); seen.add(k); prev, cur = cur, k
        loops.append(loop)
    loops.sort(key=len, reverse=True)
    return loops


def _poly_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


# ------------------------------------------------------------- reparaciones
def weld_nodes(mesh: Mesh, tol_m=5.0):
    """Suelda nodos mas cercanos que `tol_m`; el superviviente hereda la media."""
    live = mesh.live_nodes()
    if len(live) < 2:
        return "sin nodos que soldar"
    tree = cKDTree(mesh.pm[live])
    pairs = tree.query_pairs(tol_m, output_type="ndarray")
    if not len(pairs):
        return "sin nodos coincidentes"
    parent = np.arange(len(mesh.xy))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for pa, pb in live[pairs]:
        ra, rb = find(pa), find(pb)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    root = np.array([find(i) for i in range(len(mesh.xy))])
    zsum = np.zeros(len(mesh.xy)); zcnt = np.zeros(len(mesh.xy))
    np.add.at(zsum, root[live], mesh.z[live]); np.add.at(zcnt, root[live], 1.0)
    keep = zcnt > 0
    mesh.z[keep] = zsum[keep] / zcnt[keep]
    merged = (root != np.arange(len(mesh.xy))) & mesh.node_alive
    mesh.tri = root[mesh.tri]
    mesh.node_alive[merged] = False
    mesh.boundaries = [Boundary(b.kind, _dedup_seq(root[b.nodes]), b.ibtype, b.name)
                       for b in mesh.boundaries]
    mesh.touch_topology()
    n = remove_degenerate(mesh)
    return f"{int(merged.sum())} nodos soldados (<{tol_m:g} m); {n}"


def _dedup_seq(seq):
    """Quita repeticiones consecutivas de una secuencia de nodos."""
    out = [int(seq[0])] if len(seq) else []
    for v in seq[1:]:
        if int(v) != out[-1]:
            out.append(int(v))
    return np.asarray(out, np.int64)


def remove_degenerate(mesh: Mesh):
    ids, t = _live(mesh)
    bad = (t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2]) | (t[:, 0] == t[:, 2])
    dead = ~mesh.node_alive[np.clip(t, 0, len(mesh.xy) - 1)].all(axis=1)
    kill = ids[bad | dead]
    mesh.elem_alive[kill] = False
    mesh.touch_topology()
    return f"{len(kill)} elementos degenerados eliminados"


def remove_duplicate_elements(mesh: Mesh):
    ids, t = _live(mesh)
    if not len(t):
        return "malla vacia"
    ts = np.sort(t, axis=1)
    _, first = np.unique(ts, axis=0, return_index=True)
    dup = np.setdiff1d(np.arange(len(t)), first)
    mesh.elem_alive[ids[dup]] = False
    mesh.touch_topology()
    return f"{len(dup)} elementos duplicados eliminados"


def remove_tiny_area(mesh: Mesh, min_area=1.0):
    ids, t = _live(mesh)
    if not len(t):
        return "malla vacia"
    A = np.abs(signed_area(mesh.pm, t))
    kill = ids[A < min_area]
    mesh.elem_alive[kill] = False
    mesh.touch_topology()
    return f"{len(kill)} elementos de area < {min_area:g} m2 eliminados"


def fix_orientation(mesh: Mesh):
    ids, t = _live(mesh)
    if not len(t):
        return "malla vacia"
    cw = signed_area(mesh.pm, t) < 0
    k = ids[cw]
    mesh.tri[k] = mesh.tri[k][:, [0, 2, 1]]
    mesh.touch_topology()
    return f"{len(k)} elementos reorientados a CCW"


def keep_largest_component(mesh: Mesh):
    ids, t = _live(mesh)
    ue, _ = mesh.edges()
    nn = len(mesh.xy)
    if not len(ue):
        return "malla vacia"
    M = coo_matrix((np.ones(len(ue)), (ue[:, 0], ue[:, 1])), shape=(nn, nn))
    ncomp, lab = connected_components(M, directed=False)
    used = np.zeros(nn, bool); used[t.ravel()] = True
    sizes = np.bincount(lab[used], minlength=ncomp)
    if (sizes > 0).sum() <= 1:
        return "la malla ya es una sola componente"
    main = int(sizes.argmax())
    kill = ids[lab[t[:, 0]] != main]
    mesh.elem_alive[kill] = False
    mesh.node_alive[(lab != main) & used] = False
    mesh.touch_topology()
    return (f"{int((sizes > 0).sum()) - 1} componentes eliminadas "
            f"({len(kill)} elementos)")


def remove_dangling(mesh: Mesh, max_pass=10):
    """Elimina iterativamente elementos con 1 o menos vecinos."""
    total = 0
    for _ in range(max_pass):
        ids, t = _live(mesh)
        if not len(t):
            break
        ue, ce = mesh.edges()
        e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
        _, inv = np.unique(e, axis=0, return_inverse=True)
        share = (ce[inv] >= 2).reshape(3, len(t)).sum(axis=0)
        kill = ids[share <= 1]
        if not len(kill):
            break
        mesh.elem_alive[kill] = False
        mesh.touch_topology()
        total += len(kill)
    return f"{total} elementos colgantes eliminados"


def fix_nontraversable(mesh: Mesh, max_pass=8):
    """Hace la frontera recorrible: quita elementos en nodos pellizcados."""
    total = 0
    for _ in range(max_pass):
        be = mesh.boundary_edges()
        if not len(be):
            break
        bc = np.bincount(be.ravel(), minlength=len(mesh.xy))
        bad = np.flatnonzero(bc > 2)
        if not len(bad):
            break
        kill = []
        ids, t = _live(mesh)
        for i in bad:
            els = mesh.elems_of_node(int(i))
            if len(els) == 0:
                continue
            A = np.abs(signed_area(mesh.pm, mesh.tri[els]))
            kill.append(int(els[A.argmin()]))       # quita el mas pequeno
        if not kill:
            break
        mesh.elem_alive[np.asarray(kill)] = False
        mesh.touch_topology()
        total += len(kill)
    return f"{total} elementos eliminados para hacer la frontera recorrible"


def delete_boundary_slivers(mesh: Mesh, qthresh=0.05):
    """Elimina slivers que tocan la frontera (los interiores se conservan)."""
    ids, t = _live(mesh)
    if not len(t):
        return "malla vacia"
    A = signed_area(mesh.pm, t)
    L = edge_lengths(mesh.pm, t)
    q = shape_quality(A, L)
    bmask = mesh.boundary_node_mask()
    on_b = bmask[t].any(axis=1)
    kill = ids[(q < qthresh) & on_b]
    mesh.elem_alive[kill] = False
    mesh.touch_topology()
    inner = int(((q < qthresh) & ~on_b).sum())
    return f"{len(kill)} slivers de frontera eliminados ({inner} interiores conservados)"


def fill_holes(mesh: Mesh, max_nodes=30, zmin_water=3.0):
    """Rellena lazos internos pequenos totalmente sumergidos (huecos falsos)."""
    from scipy.spatial import Delaunay, QhullError
    loops = boundary_loops(mesh)
    pm = mesh.pm
    added = 0
    for loop in loops[1:]:
        if len(loop) > max_nodes or len(loop) < 3:
            continue
        idx = np.asarray(loop, np.int64)
        if not (mesh.z[idx].min() > zmin_water or len(idx) == 3):
            continue
        if len(idx) == 3:
            tris = [list(idx)]
        else:
            try:
                dt = Delaunay(pm[idx])
            except (QhullError, ValueError):
                continue
            cen = pm[idx][dt.simplices].mean(axis=1)
            keep = _point_in_poly(cen, pm[idx])
            tris = [list(idx[s]) for s in dt.simplices[keep]]
            if len(tris) < len(idx) - 2:
                continue
        for a, b, c in tris:
            mesh.tri = np.vstack([mesh.tri, [a, b, c]])
            mesh.elem_alive = np.append(mesh.elem_alive, True)
            added += 1
    if added:
        mesh.touch_topology()
        fix_orientation(mesh)
    return f"{added} elementos anadidos rellenando huecos internos"


def _point_in_poly(P, poly):
    x, y = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (y - y1) / np.where(y2 != y1, y2 - y1, np.nan) + x1
        inside ^= cond & (x < xint)
    return inside


def flip_bad_edges(mesh: Mesh, max_pass=8, max_valence=10, qgood=0.60):
    """Voltea aristas interiores cuando mejora el par de triangulos.

    Es topologico: no mueve nodos, asi que la batimetria queda intacta.
    """
    total = 0
    for _ in range(max_pass):
        ids, t = _live(mesh)
        if not len(t):
            break
        pm = mesh.pm
        q = _tri_qual(pm, t)
        emap = defaultdict(list)
        for local, tri in enumerate(t):
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i + 1) % 3])
                emap[(a, b) if a < b else (b, a)].append(local)
        val = mesh.valence()
        dead = np.zeros(len(t), bool)
        nflip = 0
        cand = sorted((k for k, v in emap.items() if len(v) == 2),
                      key=lambda k: min(q[emap[k][0]], q[emap[k][1]]))
        for (a, b) in cand:
            l1, l2 = emap[(a, b)]
            if dead[l1] or dead[l2]:
                continue
            qold = min(q[l1], q[l2])
            if qold > qgood:
                break
            c = int([v for v in t[l1] if v not in (a, b)][0])
            d = int([v for v in t[l2] if v not in (a, b)][0])
            if c == d or val[c] >= max_valence or val[d] >= max_valence:
                continue
            new = np.array([[c, d, a], [d, c, b]])
            if _tri_qual(pm, new).min() > qold + 1e-9:
                mesh.tri[ids[l1]] = new[0]
                mesh.tri[ids[l2]] = new[1]
                dead[l1] = dead[l2] = True
                val[c] += 1; val[d] += 1; val[a] -= 1; val[b] -= 1
                nflip += 1
        total += nflip
        mesh.touch_topology()
        if nflip == 0:
            break
    fix_orientation(mesh)
    return f"{total} aristas volteadas"


def _tri_qual(P, tt):
    a, b, c = P[tt[:, 0]], P[tt[:, 1]], P[tt[:, 2]]
    ar = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    l2 = ((b - a) ** 2).sum(1) + ((c - b) ** 2).sum(1) + ((a - c) ** 2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 4 * np.sqrt(3) * ar / np.where(l2 > 0, l2, np.nan)
    return np.nan_to_num(q, nan=-1.0)


def smooth_bad_regions(mesh: Mesh, qthresh=0.30, rings=3, iters=40,
                       max_disp_frac=0.25, slide_coast=True):
    """Laplaciano local alrededor de los elementos malos.

    La linea de costa se congela salvo `slide_coast`, que permite a los nodos
    de borde deslizar tangencialmente sin cambiar la forma de la costa.
    """
    from .edit import _min_quality
    ids, t = _live(mesh)
    if not len(t):
        return "malla vacia"
    pm = mesh.pm.copy()
    orig = pm.copy()
    q0 = _tri_qual(pm, t)
    nn = len(mesh.xy)
    seed = np.zeros(nn, bool)
    seed[t[q0 < qthresh].ravel()] = True
    if not seed.any():
        return f"no hay elementos con calidad < {qthresh:g}"
    indptr, data = mesh.node_adj()
    zone = seed.copy()
    for _ in range(rings):
        for i in np.flatnonzero(zone):
            zone[data[indptr[i]:indptr[i + 1]]] = True
    bmask = mesh.boundary_node_mask()
    movable = np.flatnonzero(zone & ~bmask & mesh.node_alive)

    # nodos de costa que tocan un elemento malo: deslizan sobre la tangente
    tang = {}
    if slide_coast:
        be = mesh.boundary_edges()
        bnb = defaultdict(list)
        for a, b in be:
            bnb[int(a)].append(int(b)); bnb[int(b)].append(int(a))
        for v in np.flatnonzero(seed & bmask):
            nb = bnb[int(v)]
            if len(nb) == 2:
                u = pm[nb[1]] - pm[nb[0]]
                n = float(np.hypot(u[0], u[1]))
                if n > 0:
                    tang[int(v)] = u / n

    ue, _ = mesh.edges()
    el = np.linalg.norm(pm[ue[:, 0]] - pm[ue[:, 1]], axis=1)
    hloc = np.full(nn, np.inf)
    np.minimum.at(hloc, ue[:, 0], el); np.minimum.at(hloc, ue[:, 1], el)
    maxdisp = max_disp_frac * hloc

    targets = np.concatenate([movable, np.asarray(list(tang), np.int64)]) \
        if tang else movable
    for _ in range(iters):
        changed = 0
        for v in targets:
            v = int(v)
            nb = data[indptr[v]:indptr[v + 1]]
            if len(nb) < 3:
                continue
            els = mesh.elems_of_node(v)
            qcur = _min_quality(pm, mesh.tri[els])
            if qcur > 0.85:
                continue
            old = pm[v].copy()
            step = pm[nb].mean(axis=0) - old
            if v in tang:
                u = tang[v]
                step = float(step @ u) * u
            n = float(np.hypot(step[0], step[1]))
            if n > maxdisp[v]:
                step *= maxdisp[v] / n
            best, bq = None, qcur
            for frac in (1.0, 0.6, 0.3, 0.1):
                pm[v] = old + frac * step
                qn = _min_quality(pm, mesh.tri[els])
                if qn > bq + 1e-12:
                    best, bq = pm[v].copy(), qn
            pm[v] = best if best is not None else old
            if best is not None:
                changed += 1
        if changed == 0:
            break

    moved = np.flatnonzero(np.linalg.norm(pm - orig, axis=1) > 1e-6)
    if not len(moved):
        return "el suavizado no encontro mejoras"
    mesh.xy[moved] = mesh.to_degrees(pm[moved])
    mesh.touch_geometry()
    q1 = _tri_qual(mesh.pm, t)
    return (f"{len(moved)} nodos movidos; calidad minima "
            f"{q0.min():.4f} -> {q1.min():.4f}; elementos q<{qthresh:g}: "
            f"{int((q0 < qthresh).sum())} -> {int((q1 < qthresh).sum())}")


def reinterpolate_depths(mesh: Mesh, ref_xy, ref_z, ids=None):
    """Reinterpola la batimetria en ciertos nodos desde una malla de referencia."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    pm_ref = mesh.to_meters(ref_xy)
    lin = LinearNDInterpolator(pm_ref, ref_z)
    nea = NearestNDInterpolator(pm_ref, ref_z)
    ids = mesh.live_nodes() if ids is None else np.atleast_1d(ids)
    p = mesh.pm[ids]
    zi = lin(p)
    bad = ~np.isfinite(zi)
    if bad.any():
        zi[bad] = nea(p[bad])
    mesh.z[ids] = zi
    mesh.touch_geometry()
    return f"batimetria reinterpolada en {len(ids)} nodos"


def renumber_rcm(mesh: Mesh):
    """Renumera con Cuthill-McKee inverso: reduce el ancho de banda de ADCIRC."""
    xy, z, tri, bnds = mesh.compacted()
    nn = len(xy)
    e = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    ue = np.unique(e, axis=0)
    M = coo_matrix((np.ones(len(ue)), (ue[:, 0], ue[:, 1])), shape=(nn, nn)).tocsr()
    M = M + M.T
    perm = reverse_cuthill_mckee(M, symmetric_mode=True)
    inv = np.empty(nn, np.int64)
    inv[perm] = np.arange(nn)

    def bandwidth(tt):
        return int((tt.max(axis=1) - tt.min(axis=1)).max())

    bw0 = bandwidth(tri)
    tri2 = inv[tri]
    bw1 = bandwidth(tri2)
    mesh.xy = xy[perm]; mesh.z = z[perm]; mesh.tri = tri2
    mesh.node_alive = np.ones(nn, bool)
    mesh.elem_alive = np.ones(len(tri2), bool)
    mesh.boundaries = [Boundary(b.kind, inv[b.nodes], b.ibtype, b.name, b.extra)
                       for b in bnds]
    mesh.touch_topology()
    return f"ancho de banda {bw0} -> {bw1} ({100 * (1 - bw1 / max(bw0, 1)):.0f}% menos)"


def compact_mesh(mesh: Mesh):
    """Elimina fisicamente lo muerto y reindexa."""
    n0, e0 = len(mesh.xy), len(mesh.tri)
    xy, z, tri, bnds = mesh.compacted()
    mesh.xy, mesh.z, mesh.tri = xy, z, tri
    mesh.node_alive = np.ones(len(xy), bool)
    mesh.elem_alive = np.ones(len(tri), bool)
    mesh.boundaries = bnds
    mesh.touch_topology()
    return f"compactado: {n0}->{len(xy)} nodos, {e0}->{len(tri)} elementos"


# ------------------------------------------------------------- fronteras auto
def auto_boundaries(mesh: Mesh, min_edge_m=3000.0, min_depth=200.0,
                    land_ibtype=20, island_ibtype=21):
    """Detecta frontera abierta (oceano profundo y grueso), tierra e islas."""
    loops = boundary_loops(mesh)
    if not loops:
        return "no se encontro frontera"
    pm, z = mesh.pm, mesh.z
    o = np.asarray(loops[0], np.int64)
    n = len(o)
    seg = np.linalg.norm(pm[o] - pm[np.roll(o, -1)], axis=1)
    nodelen = np.maximum(seg, np.roll(seg, 1))
    ocean = (nodelen > min_edge_m) & (z[o] > min_depth)

    runs, ext, i = [], np.concatenate([ocean, ocean]), 0
    while i < 2 * n:
        if ext[i]:
            j = i
            while j < 2 * n and ext[j]:
                j += 1
            if j - i <= n:
                runs.append((i, j - i))
            i = j
        else:
            i += 1
    if not runs:
        return ("no se detecto frontera oceanica: baja los umbrales de "
                "profundidad o de tamano de arista")
    runs.sort(key=lambda r: -r[1])
    st, ln = runs[0]
    core = ln
    while z[o[(st - 1) % n]] > 0 and ln < n - 3:
        st -= 1; ln += 1
    while z[o[(st + ln) % n]] > 0 and ln < n - 3:
        ln += 1
    open_nodes = o[[(st + k) % n for k in range(ln)]]
    land_nodes = o[[(st + ln - 1 + k) % n for k in range(n - ln + 2)]]
    if _poly_area(pm[o]) < 0:
        open_nodes = open_nodes[::-1]; land_nodes = land_nodes[::-1]

    bnds = [Boundary(OCEAN, open_nodes, None, "Frontera abierta"),
            Boundary(LAND, land_nodes, land_ibtype, "Tierra firme")]
    nisl = 0
    for loop in loops[1:]:
        li = np.asarray(loop, np.int64)
        if len(li) < 3:
            continue
        if _poly_area(pm[li]) > 0:
            li = li[::-1]                       # islas en sentido horario
        bnds.append(Boundary(ISLAND, li, island_ibtype, f"Isla {nisl + 1}"))
        nisl += 1
    mesh.boundaries = bnds
    mesh.touch_topology()
    return (f"frontera abierta: {len(open_nodes)} nodos (nucleo profundo {core}), "
            f"tierra: {len(land_nodes)}, islas: {nisl}")


# ------------------------------------------------------- pipeline automatico
PIPELINE = [
    ("Soldar nodos coincidentes", lambda m: weld_nodes(m, 5.0)),
    ("Eliminar degenerados", remove_degenerate),
    ("Eliminar duplicados", remove_duplicate_elements),
    ("Eliminar area nula", lambda m: remove_tiny_area(m, 1.0)),
    ("Orientar CCW", fix_orientation),
    ("Componente principal", keep_largest_component),
    ("Frontera recorrible", fix_nontraversable),
    ("Elementos colgantes", remove_dangling),
    ("Rellenar huecos falsos", fill_holes),
    ("Suavizado local", lambda m: smooth_bad_regions(m, 0.30)),
    ("Voltear aristas", flip_bad_edges),
    ("Slivers de frontera", lambda m: delete_boundary_slivers(m, 0.05)),
    ("Elementos colgantes (2a)", remove_dangling),
    ("Compactar", compact_mesh),
]


def run_pipeline(mesh: Mesh, steps=None, progress=None):
    out = []
    for i, (name, fn) in enumerate(PIPELINE):
        if steps is not None and name not in steps:
            continue
        if progress:
            progress(i, len(PIPELINE), name)
        try:
            msg = fn(mesh)
        except Exception as exc:                 # una etapa no debe tumbar todo
            msg = f"ERROR: {exc}"
        out.append(f"[{name}] {msg}")
    return "\n".join(out)

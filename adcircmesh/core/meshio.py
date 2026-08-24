"""Lectura/escritura de mallas: fort.14 (ADCIRC) y .2dm (SMS Aquaveo)."""
from __future__ import annotations

import numpy as np

from .mesh import Mesh, Boundary, OCEAN, LAND, ISLAND

# IBTYPE de tierra que ADCIRC considera islas cerradas
ISLAND_IBTYPES = {1, 11, 21}
# IBTYPE que llevan datos extra por nodo en el fort.14
WEIR_IBTYPES = {3, 13, 23, 4, 24, 5, 25}


# ------------------------------------------------------------------ fort.14
def read_fort14(path, progress=None):
    """Lee un fort.14 completo, incluidas las fronteras. Devuelve un `Mesh`."""
    with open(path, "r") as f:
        raw = f.read()
    lines = raw.splitlines()
    title = lines[0].strip()
    ne, nn = (int(v) for v in lines[1].split()[:2])
    if progress:
        progress(f"leyendo {nn} nodos / {ne} elementos")

    # nodos: columnas 1..3 de las siguientes nn lineas
    blk = "\n".join(lines[2:2 + nn])
    arr = np.fromstring(blk.replace("D", "E"), sep=" ")
    arr = arr.reshape(nn, -1)
    xy = arr[:, 1:3].astype(float)
    z = arr[:, 3].astype(float)

    # elementos: id, nnodes(=3), n1, n2, n3
    blk = "\n".join(lines[2 + nn:2 + nn + ne])
    earr = np.fromstring(blk, sep=" ", dtype=float).reshape(ne, -1)
    tri = earr[:, 2:5].astype(np.int64) - 1

    boundaries = _read_boundaries(lines[2 + nn + ne:])
    m = Mesh(xy, z, tri, boundaries, title)
    return m


def _read_boundaries(rest):
    """Parsea las secciones NOPE/NBOU al final del fort.14."""
    rows = [ln for ln in rest if ln.strip()]
    out = []
    if not rows:
        return out
    i = 0

    def first_int(k):
        return int(float(rows[k].split()[0]))

    try:
        nope = first_int(i); i += 1
        i += 1                                    # NETA (redundante)
        for _ in range(nope):
            nv = first_int(i); i += 1
            nodes = [first_int(i + k) - 1 for k in range(nv)]
            i += nv
            out.append(Boundary(OCEAN, nodes, None, "Frontera abierta"))

        nbou = first_int(i); i += 1
        i += 1                                    # NVEL (redundante)
        for _ in range(nbou):
            tk = rows[i].split(); i += 1
            nv = int(float(tk[0]))
            ibt = int(float(tk[1])) if len(tk) > 1 else 0
            nodes, extra = [], []
            for k in range(nv):
                parts = rows[i + k].split()
                nodes.append(int(float(parts[0])) - 1)
                extra.append(" ".join(parts[1:]) if len(parts) > 1 else "")
            i += nv
            kind = ISLAND if ibt in ISLAND_IBTYPES else LAND
            has_extra = ibt in WEIR_IBTYPES and any(extra)
            out.append(Boundary(kind, nodes, ibt, "",
                                extra if has_extra else None))
    except (IndexError, ValueError):
        pass                                       # frontera truncada: usa lo leido
    return out


def write_fort14(path, mesh: Mesh, progress=None):
    xy, z, tri, bnds = mesh.compacted()
    if progress:
        progress(f"escribiendo {len(xy)} nodos / {len(tri)} elementos")

    nid = np.arange(1, len(xy) + 1)
    node_block = np.column_stack([nid, xy[:, 0], xy[:, 1], z])
    eid = np.arange(1, len(tri) + 1)
    elem_block = np.column_stack([eid, np.full(len(tri), 3), tri + 1])

    with open(path, "w") as f:
        f.write(f"{mesh.title}\n")
        f.write(f"{len(tri)} {len(xy)}\n")
        np.savetxt(f, node_block, fmt="%d %.8f %.8f %.6f")
        np.savetxt(f, elem_block, fmt="%d %d %d %d %d")
        f.write(_boundary_block(bnds))


def _boundary_block(bnds):
    ocean = [b for b in bnds if b.kind == OCEAN]
    land = [b for b in bnds if b.kind != OCEAN]
    L = []
    L.append(f"{len(ocean)}   ! NOPE: numero de fronteras abiertas")
    L.append(f"{sum(len(b) for b in ocean)}   ! NETA: total nodos frontera abierta")
    for k, b in enumerate(ocean, 1):
        L.append(f"{len(b)}   ! NVDLL({k})")
        L += [str(n + 1) for n in b.nodes]
    L.append(f"{len(land)}   ! NBOU: numero de fronteras de tierra")
    L.append(f"{sum(len(b) for b in land)}   ! NVEL: total nodos frontera tierra")
    for k, b in enumerate(land, 1):
        ibt = 21 if (b.ibtype is None and b.kind == ISLAND) else (b.ibtype or 20)
        L.append(f"{len(b)} {ibt}   ! NVELL({k}) IBTYPE({k})")
        if b.extra:
            L += [f"{n + 1} {e}".rstrip() for n, e in zip(b.nodes, b.extra)]
        else:
            L += [str(n + 1) for n in b.nodes]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- 2dm
def read_2dm(path, flip_z=True):
    """Lee un .2dm de SMS.  `flip_z`: 2dm guarda elevacion, ADCIRC profundidad."""
    nid, nxy, nz = [], [], []
    tris, nodestrings, cur = [], [], []
    with open(path) as f:
        for ln in f:
            tk = ln.split()
            if not tk:
                continue
            c = tk[0]
            if c == "ND":
                nid.append(int(tk[1]))
                nxy.append((float(tk[2]), float(tk[3])))
                nz.append(float(tk[4]))
            elif c in ("E3T",):
                tris.append((int(tk[2]), int(tk[3]), int(tk[4])))
            elif c == "E4Q":                      # se divide el quad en 2 tri
                a, b, cc, d = (int(v) for v in tk[2:6])
                tris.append((a, b, cc)); tris.append((a, cc, d))
            elif c == "NS":
                for v in tk[1:]:
                    v = int(float(v))
                    cur.append(abs(v) - 1)
                    if v < 0:                     # negativo = fin del nodestring
                        nodestrings.append(cur); cur = []
    if cur:
        nodestrings.append(cur)

    maxid = max(nid)
    remap = -np.ones(maxid + 1, np.int64)
    remap[np.asarray(nid)] = np.arange(len(nid))
    xy = np.asarray(nxy, float)
    z = np.asarray(nz, float)
    if flip_z:
        z = -z
    tri = remap[np.asarray(tris, np.int64)]
    bnds = [Boundary(LAND, remap[np.asarray(ns)], 20, f"NS{k+1}")
            for k, ns in enumerate(nodestrings)]
    from pathlib import Path
    return Mesh(xy, z, tri, bnds, Path(path).stem)


def write_2dm(path, mesh: Mesh, flip_z=True):
    xy, z, tri, bnds = mesh.compacted()
    zz = -z if flip_z else z
    with open(path, "w") as f:
        f.write("MESH2D\n")
        for k, t in enumerate(tri + 1, 1):
            f.write(f"E3T {k} {t[0]} {t[1]} {t[2]} 1\n")
        for k, ((x, y), d) in enumerate(zip(xy, zz), 1):
            f.write(f"ND {k} {x:.8f} {y:.8f} {d:.6f}\n")
        for b in bnds:
            n = b.nodes + 1
            for s in range(0, len(n), 10):
                chunk = list(n[s:s + 10])
                if s + 10 >= len(n):
                    chunk[-1] = -chunk[-1]
                f.write("NS " + " ".join(str(v) for v in chunk) + "\n")


# --------------------------------------------------------------- despachador
def load_mesh(path, progress=None):
    p = str(path).lower()
    if p.endswith(".2dm"):
        return read_2dm(path)
    return read_fort14(path, progress)


def save_mesh(path, mesh, progress=None):
    p = str(path).lower()
    if p.endswith(".2dm"):
        return write_2dm(path, mesh)
    return write_fort14(path, mesh, progress)

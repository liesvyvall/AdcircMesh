"""Control de calidad de mallas para ADCIRC / SWAN.

Cada chequeo devuelve no solo un conteo sino los INDICES infractores, para que
la interfaz pueda resaltarlos y hacer zoom sobre ellos (como el Mesh Quality de
SMS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .mesh import Mesh, G, OCEAN

ERROR, WARN, INFO, OK = "error", "warning", "info", "ok"
NODE, ELEM, EDGE, NONE = "node", "element", "edge", "none"


@dataclass
class Check:
    key: str
    label: str
    count: int
    severity: str
    kind: str = NONE
    ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    detail: str = ""
    group: str = "General"


@dataclass
class QCReport:
    checks: list
    fields: dict          # nombre -> (valores, tipo NODE/ELEM)
    stats: dict
    elem_ids: np.ndarray  # ids globales de los elementos vivos analizados

    @property
    def n_errors(self):
        return sum(1 for c in self.checks if c.severity == ERROR and c.count)

    @property
    def n_warnings(self):
        return sum(1 for c in self.checks if c.severity == WARN and c.count)

    def by_key(self, key):
        for c in self.checks:
            if c.key == key:
                return c
        return None


# ---------------------------------------------------------------- primitivas
def signed_area(pm, t):
    a, b, c = pm[t[:, 0]], pm[t[:, 1]], pm[t[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                  (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def edge_lengths(pm, t):
    L = np.empty((len(t), 3))
    for k, (i, j) in enumerate([(1, 2), (2, 0), (0, 1)]):
        L[:, k] = np.linalg.norm(pm[t[:, i]] - pm[t[:, j]], axis=1)
    return L


def angles_deg(L):
    a, b, c = L[:, 0], L[:, 1], L[:, 2]
    def ang(o, p, q):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (p ** 2 + q ** 2 - o ** 2) / (2 * p * q)
        return np.degrees(np.arccos(np.clip(v, -1, 1)))
    return np.column_stack([ang(a, b, c), ang(b, c, a), ang(c, a, b)])


def shape_quality(A, L):
    """Razon de forma normalizada: 1 = equilatero, 0 = degenerado."""
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 4 * np.sqrt(3) * np.abs(A) / (L ** 2).sum(axis=1)
    return np.nan_to_num(q, nan=0.0)


def element_fields(mesh: Mesh):
    """Campos geometricos por elemento vivo (para colorear y para QC)."""
    ids, t = mesh.live_elems()
    pm = mesh.pm
    A = signed_area(pm, t)
    L = edge_lengths(pm, t)
    ang = angles_deg(L)
    q = shape_quality(A, L)
    amin = np.nan_to_num(ang.min(axis=1), nan=0.0)
    amax = np.nan_to_num(ang.max(axis=1), nan=180.0)
    zc = mesh.z[t].mean(axis=1)
    hmin = L.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dt = hmin / np.sqrt(G * np.clip(zc, 0.1, None))
    return dict(ids=ids, tri=t, area=np.abs(A), signed=A, L=L, ang=ang,
                qual=q, amin=amin, amax=amax, depth_c=zc, hmin=hmin,
                dt_cfl=np.nan_to_num(dt, nan=0.0))


# ------------------------------------------------------------------ chequeos
def run_qc(mesh: Mesh, dt_target=1.0, min_angle=30.0, max_valence=10,
           min_depth=1.0, gradation=1.5, progress=None):
    """Ejecuta la bateria completa. Devuelve un `QCReport`."""
    def step(msg):
        if progress:
            progress(msg)

    checks = []
    add = checks.append
    nn = len(mesh.xy)
    ef = element_fields(mesh)
    ids, t = ef["ids"], ef["tri"]
    ne = len(t)
    pm = mesh.pm
    alive_nodes = mesh.live_nodes()

    def sev(count, s=ERROR):
        return s if count else OK

    # ---------------------------------------------------- 1. integridad
    step("integridad topologica")
    bad = np.flatnonzero((t < 0).any(1) | (t >= nn).any(1))
    add(Check("idx_range", "Indices de nodo fuera de rango", len(bad),
              sev(len(bad)), ELEM, ids[bad], group="Integridad"))

    degen = np.flatnonzero((t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2]) | (t[:, 0] == t[:, 2]))
    add(Check("degenerate", "Elementos con nodos repetidos", len(degen),
              sev(len(degen)), ELEM, ids[degen], group="Integridad"))

    dead_ref = np.flatnonzero(~mesh.node_alive[np.clip(t, 0, nn - 1)].all(1))
    add(Check("dead_ref", "Elementos que usan nodos borrados", len(dead_ref),
              sev(len(dead_ref)), ELEM, ids[dead_ref], group="Integridad"))

    # nodos coincidentes
    step("nodos coincidentes")
    key = np.round(mesh.xy[alive_nodes], 6)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    dupmask = cnt[inv] > 1
    add(Check("dup_nodes", "Nodos coincidentes (< 1e-6 grados)", int(dupmask.sum()),
              sev(int(dupmask.sum()), WARN), NODE, alive_nodes[dupmask],
              group="Integridad"))

    ts = np.sort(t, axis=1)
    _, inv_e, cnt_e = np.unique(ts, axis=0, return_inverse=True, return_counts=True)
    dupe = np.flatnonzero(cnt_e[inv_e] > 1)
    add(Check("dup_elems", "Elementos duplicados", len(dupe), sev(len(dupe)),
              ELEM, ids[dupe], group="Integridad"))

    used = np.zeros(nn, bool)
    if ne:
        used[t.ravel()] = True
    orphan = np.flatnonzero(mesh.node_alive & ~used)
    add(Check("orphans", "Nodos huerfanos (sin elemento)", len(orphan),
              sev(len(orphan), WARN), NODE, orphan, group="Integridad"))

    # ---------------------------------------------------- 2. topologia
    step("topologia de aristas")
    ue, ce = mesh.edges()
    nonman = ue[ce > 2]
    nm_nodes = np.unique(nonman.ravel()) if len(nonman) else np.zeros(0, np.int64)
    add(Check("nonmanifold", "Aristas no-manifold (>2 elementos)", len(nonman),
              sev(len(nonman)), NODE, nm_nodes, group="Topologia"))

    bnd_e = ue[ce == 1]
    bmask = mesh.boundary_node_mask()
    if len(bnd_e):
        bc = np.bincount(bnd_e.ravel(), minlength=nn)
        nontrav = np.flatnonzero(bc > 2)
    else:
        nontrav = np.zeros(0, np.int64)
    add(Check("nontraversable", "Nodos de frontera no recorribles (>2 aristas de borde)",
              len(nontrav), sev(len(nontrav)), NODE, nontrav, group="Topologia"))

    # componentes desconectadas
    step("componentes conectadas")
    if len(ue):
        M = coo_matrix((np.ones(len(ue)), (ue[:, 0], ue[:, 1])), shape=(nn, nn))
        ncomp, lab = connected_components(M, directed=False)
        sizes = np.bincount(lab[used], minlength=ncomp)
        main = int(sizes.argmax())
        stray = np.flatnonzero(used & (lab != main))
        nsub = int((sizes > 0).sum() - 1)
    else:
        lab = np.zeros(nn, np.int64); stray = np.zeros(0, np.int64); nsub = 0
    add(Check("components", "Componentes desconectadas", nsub, sev(nsub),
              NODE, stray, group="Topologia",
              detail="ADCIRC requiere un dominio de una sola pieza"))

    # elementos colgantes: comparten <=1 arista con otro elemento
    step("elementos colgantes")
    if ne:
        e_all = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
        _, inv_ed = np.unique(e_all, axis=0, return_inverse=True)
        share = (ce[inv_ed] >= 2).reshape(3, ne).sum(axis=0)
        dangling = np.flatnonzero(share <= 1)
    else:
        dangling = np.zeros(0, np.int64)
    add(Check("dangling", "Elementos colgantes (<=1 vecino)", len(dangling),
              sev(len(dangling), WARN), ELEM, ids[dangling], group="Topologia"))

    # valencia
    val = mesh.valence()
    hi = np.flatnonzero((val > max_valence) & used)
    add(Check("valence_hi", f"Nodos con valencia > {max_valence}", len(hi),
              sev(len(hi), WARN), NODE, hi, group="Topologia"))
    lo = np.flatnonzero((val < 3) & used)
    add(Check("valence_lo", "Nodos con valencia < 3", len(lo),
              sev(len(lo), WARN), NODE, lo, group="Topologia"))
    inner_lo = np.flatnonzero((val < 4) & used & ~bmask)
    add(Check("valence_inner", "Nodos interiores con valencia < 4", len(inner_lo),
              sev(len(inner_lo), INFO), NODE, inner_lo, group="Topologia"))

    # ---------------------------------------------------- 3. geometria
    step("geometria de elementos")
    cw = np.flatnonzero(ef["signed"] < 0)
    add(Check("orientation", "Elementos horarios (CW) - ADCIRC exige CCW", len(cw),
              sev(len(cw)), ELEM, ids[cw], group="Geometria"))

    tiny = np.flatnonzero(ef["area"] < 1.0)
    add(Check("zero_area", "Elementos de area < 1 m2", len(tiny), sev(len(tiny)),
              ELEM, ids[tiny], group="Geometria"))

    q = ef["qual"]
    sl = np.flatnonzero(q < 0.10)
    add(Check("sliver", "Slivers (calidad < 0.10)", len(sl), sev(len(sl)),
              ELEM, ids[sl], group="Geometria"))
    poor = np.flatnonzero(q < 0.30)
    add(Check("poor_shape", "Elementos de forma pobre (calidad < 0.30)", len(poor),
              sev(len(poor), WARN), ELEM, ids[poor], group="Geometria"))

    amin, amax = ef["amin"], ef["amax"]
    a10 = np.flatnonzero(amin < 10)
    add(Check("angle_10", "Angulo minimo < 10 grados", len(a10), sev(len(a10)),
              ELEM, ids[a10], group="Geometria"))
    am = np.flatnonzero(amin < min_angle)
    add(Check("angle_min", f"Angulo minimo < {min_angle:g} grados", len(am),
              sev(len(am), WARN), ELEM, ids[am], group="Geometria"))
    ax = np.flatnonzero(amax > 150)
    add(Check("angle_max", "Angulo maximo > 150 grados", len(ax),
              sev(len(ax), WARN), ELEM, ids[ax], group="Geometria"))

    # elementos con los 3 nodos sobre la frontera
    if ne:
        allb = np.flatnonzero(bmask[t].all(axis=1))
    else:
        allb = np.zeros(0, np.int64)
    add(Check("bnd_tri", "Elementos con sus 3 nodos en la frontera", len(allb),
              sev(len(allb), WARN), ELEM, ids[allb], group="Geometria",
              detail="Bloquean el flujo: tipicos slivers de costa"))

    # gradacion: razon entre la arista mas larga y la mas corta incidentes al nodo
    step("gradacion de tamano")
    if len(ue):
        el = np.linalg.norm(pm[ue[:, 0]] - pm[ue[:, 1]], axis=1)
        emin = np.full(nn, np.inf); emax = np.zeros(nn)
        np.minimum.at(emin, ue[:, 0], el); np.minimum.at(emin, ue[:, 1], el)
        np.maximum.at(emax, ue[:, 0], el); np.maximum.at(emax, ue[:, 1], el)
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = np.where(np.isfinite(emin) & (emin > 0), emax / emin, 1.0)
        bad_g = np.flatnonzero((grad > gradation) & used)
    else:
        grad = np.ones(nn); bad_g = np.zeros(0, np.int64)
    add(Check("gradation", f"Nodos con gradacion > {gradation:g}", len(bad_g),
              sev(len(bad_g), WARN), NODE, bad_g, group="Geometria",
              detail="Cambios bruscos de resolucion degradan la solucion"))

    # ---------------------------------------------------- 4. ADCIRC / SWAN
    step("criterios ADCIRC/SWAN")
    dt = ef["dt_cfl"]
    cfl = np.flatnonzero(dt < dt_target)
    add(Check("cfl", f"Elementos que violan CFL con dt = {dt_target:g} s", len(cfl),
              sev(len(cfl), WARN), ELEM, ids[cfl], group="ADCIRC/SWAN",
              detail=f"dt_max global = {dt.min():.4f} s"))

    zdry = np.flatnonzero((mesh.z <= 0) & mesh.node_alive & used)
    add(Check("dry", "Nodos con profundidad <= 0 (sobre el datum)", len(zdry),
              sev(len(zdry), WARN), NODE, zdry, group="ADCIRC/SWAN",
              detail="Requieren mojado/secado activado (NOLIFA>=2)"))
    zshallow = np.flatnonzero((mesh.z > 0) & (mesh.z < min_depth) & mesh.node_alive & used)
    add(Check("shallow", f"Nodos con profundidad < {min_depth:g} m", len(zshallow),
              sev(len(zshallow), INFO), NODE, zshallow, group="ADCIRC/SWAN"))
    zdeep = np.flatnonzero((mesh.z < -10) & mesh.node_alive & used)
    add(Check("above_datum", "Nodos con profundidad < -10 m", len(zdeep),
              sev(len(zdeep), WARN), NODE, zdeep, group="ADCIRC/SWAN"))
    znan = np.flatnonzero(~np.isfinite(mesh.z) & mesh.node_alive)
    add(Check("nan_depth", "Nodos con profundidad NaN/inf", len(znan),
              sev(len(znan)), NODE, znan, group="ADCIRC/SWAN"))

    # gradiente batimetrico por arista (relevante para SWAN)
    if len(ue):
        dz = np.abs(mesh.z[ue[:, 0]] - mesh.z[ue[:, 1]])
        zmean = np.maximum(np.abs(mesh.z[ue]).mean(axis=1), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = dz / zmean
        steep = ue[rel > 1.0]
        steep_n = np.unique(steep.ravel()) if len(steep) else np.zeros(0, np.int64)
    else:
        steep_n = np.zeros(0, np.int64)
    add(Check("bathy_jump", "Saltos batimetricos > 100% entre nodos vecinos",
              len(steep_n), sev(len(steep_n), INFO), NODE, steep_n,
              group="ADCIRC/SWAN", detail="SWAN puede volverse inestable"))

    # ---------------------------------------------------- 5. fronteras
    step("fronteras declaradas")
    checks += _boundary_checks(mesh, bmask, used, nn)

    # ---------------------------------------------------- estadisticas
    stats = {}
    if ne:
        for name, arr, unit in [
            ("Calidad de forma", q, ""), ("Angulo minimo", amin, "deg"),
            ("Angulo maximo", amax, "deg"), ("Arista minima", ef["hmin"], "m"),
            ("Area", ef["area"], "m2"), ("dt maximo (CFL)", dt, "s"),
        ]:
            stats[name] = dict(min=float(np.nanmin(arr)), p1=float(np.nanpercentile(arr, 1)),
                               med=float(np.nanmedian(arr)), p99=float(np.nanpercentile(arr, 99)),
                               max=float(np.nanmax(arr)), unit=unit)
        L = ef["L"]
        stats["Resolucion (aristas)"] = dict(
            min=float(L.min()), p1=float(np.percentile(L, 1)), med=float(np.median(L)),
            p99=float(np.percentile(L, 99)), max=float(L.max()), unit="m")
    zz = mesh.z[mesh.node_alive]
    if len(zz):
        stats["Profundidad"] = dict(min=float(zz.min()), p1=float(np.percentile(zz, 1)),
                                    med=float(np.median(zz)), p99=float(np.percentile(zz, 99)),
                                    max=float(zz.max()), unit="m")

    fields = {
        "Profundidad": (mesh.z, NODE),
        "Calidad de forma": (q, ELEM),
        "Angulo minimo": (amin, ELEM),
        "Angulo maximo": (amax, ELEM),
        "Area del elemento": (ef["area"], ELEM),
        "Arista minima": (ef["hmin"], ELEM),
        "dt maximo (CFL)": (dt, ELEM),
        "Gradacion": (grad, NODE),
        "Valencia": (val.astype(float), NODE),
    }
    return QCReport(checks, fields, stats, ids)


def _boundary_checks(mesh: Mesh, bmask, used, nn):
    out = []
    bnds = mesh.boundaries
    ocean = [b for b in bnds if b.kind == OCEAN]
    if not bnds:
        out.append(Check("no_bnd", "No hay fronteras definidas", 1, ERROR, NONE,
                         group="Fronteras",
                         detail="ADCIRC no puede correr sin las secciones NOPE/NBOU"))
        return out
    out.append(Check("no_open", "Sin frontera abierta (NOPE = 0)", 0 if ocean else 1,
                     OK if ocean else ERROR, NONE, group="Fronteras",
                     detail="Sin ella no se puede forzar la marea"))

    allb = np.concatenate([b.nodes for b in bnds]) if bnds else np.zeros(0, np.int64)
    oob = allb[(allb < 0) | (allb >= nn)]
    out.append(Check("bnd_oob", "Nodos de frontera fuera de rango", len(oob),
                     ERROR if len(oob) else OK, NONE, group="Fronteras"))
    inr = allb[(allb >= 0) & (allb < nn)]

    notb = inr[~bmask[inr]]
    out.append(Check("bnd_interior", "Nodos declarados que NO estan en el borde real",
                     len(np.unique(notb)), WARN if len(notb) else OK, NODE,
                     np.unique(notb), group="Fronteras"))

    dead = inr[~mesh.node_alive[inr]]
    out.append(Check("bnd_dead", "Nodos de frontera borrados de la malla", len(dead),
                     ERROR if len(dead) else OK, NODE, np.unique(dead), group="Fronteras"))

    real_b = np.flatnonzero(bmask & used)
    missing = np.setdiff1d(real_b, inr)
    out.append(Check("bnd_missing", "Nodos del borde real sin declarar", len(missing),
                     WARN if len(missing) else OK, NODE, missing, group="Fronteras",
                     detail="ADCIRC los trata como pared, puede no ser lo deseado"))

    u, c = np.unique(inr, return_counts=True)
    over = u[c > 1]
    out.append(Check("bnd_overlap", "Nodos en mas de una frontera", len(over),
                     WARN if len(over) else OK, NODE, over, group="Fronteras"))
    return out

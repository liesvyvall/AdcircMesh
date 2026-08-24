"""Vista de malla: pyqtgraph para la geometria vectorial + un raster Agg para
el relleno de color.

El wireframe completo (368k aristas) se dibuja en ~25 ms con un unico
QPainterPath, asi que no hace falta nivel de detalle.  El relleno de color, en
cambio, se rasteriza fuera de pantalla con matplotlib/Agg y se muestra como
imagen; se regenera solo cuando la vista se estabiliza.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False,
                    background="w", foreground="#3c4250")

# modos de herramienta
NAV = "nav"
SEL_NODE = "sel_node"
SEL_ELEM = "sel_elem"
MOVE_NODE = "move_node"
ADD_NODE = "add_node"
DEL_NODE = "del_node"
DEL_ELEM = "del_elem"
SPLIT_EDGE = "split_edge"
SWAP_EDGE = "swap_edge"
CREATE_ELEM = "create_elem"
MERGE_NODES = "merge_nodes"
NODESTRING = "nodestring"

TOOL_HELP = {
    NAV: "Navegar: arrastra para desplazar, rueda para zoom",
    SEL_NODE: "Seleccionar nodos: clic o arrastra un rectangulo. Shift suma, Ctrl resta",
    SEL_ELEM: "Seleccionar elementos: clic o arrastra un rectangulo. Shift suma, Ctrl resta",
    MOVE_NODE: "Mover nodo: arrastra un nodo a su nueva posicion",
    ADD_NODE: "Anadir nodo: clic dentro de un elemento para subdividirlo en 3",
    DEL_NODE: "Borrar nodo: clic sobre el nodo (elimina sus elementos)",
    DEL_ELEM: "Borrar elemento: clic dentro del elemento",
    SPLIT_EDGE: "Dividir arista: clic sobre la arista",
    SWAP_EDGE: "Voltear arista: clic sobre la diagonal interior",
    CREATE_ELEM: "Crear elemento: clic en 3 nodos consecutivos",
    MERGE_NODES: "Fusionar nodos: clic en el nodo que se conserva y luego en el que desaparece",
    NODESTRING: "Nodestring: clic en nodos de frontera consecutivos, Enter para cerrar",
}

COL_WIRE = (152, 160, 172)          # gris medio: legible sin tapar el relleno
COL_BND = (70, 78, 92)              # borde real de la malla
COL_OCEAN = (0, 122, 204)           # nodestring de frontera abierta
COL_LAND = (224, 108, 16)           # nodestring de tierra
COL_ISLAND = (17, 138, 68)          # nodestring de isla
COL_SEL = (124, 58, 237)            # violeta: no choca con ninguna frontera
COL_HL = (214, 25, 55)              # rojo para los casos del control de calidad
COL_NODE = (72, 82, 98)


class EditViewBox(pg.ViewBox):
    """ViewBox que cede el boton izquierdo a la herramienta activa."""

    sigToolPress = QtCore.Signal(object, object)     # (x, y), modifiers
    sigToolDrag = QtCore.Signal(object, object, object)  # start, cur, state
    sigToolClick = QtCore.Signal(object, object)
    sigCursorMoved = QtCore.Signal(object)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.tool = NAV
        self._drag_start = None

    def mouseDragEvent(self, ev, axis=None):
        if self.tool == NAV or ev.button() != QtCore.Qt.LeftButton:
            return super().mouseDragEvent(ev, axis)
        ev.accept()
        p = self.mapToView(ev.pos())
        cur = (p.x(), p.y())
        if ev.isStart():
            s = self.mapToView(ev.buttonDownPos())
            self._drag_start = (s.x(), s.y())
            state = "start"
        elif ev.isFinish():
            state = "finish"
        else:
            state = "move"
        self.sigToolDrag.emit(self._drag_start, cur, (state, ev.modifiers()))

    def mouseClickEvent(self, ev):
        if self.tool != NAV and ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            p = self.mapToView(ev.pos())
            self.sigToolClick.emit((p.x(), p.y()), ev.modifiers())
            return
        super().mouseClickEvent(ev)

    def hoverEvent(self, ev):
        if not ev.isExit():
            p = self.mapToView(ev.pos())
            self.sigCursorMoved.emit((p.x(), p.y()))
        super().hoverEvent(ev)


class MeshView(QtWidgets.QWidget):
    """Lienzo principal de la malla."""

    sigStatus = QtCore.Signal(str)
    sigCursor = QtCore.Signal(float, float)
    sigToolAction = QtCore.Signal(str, object, object)   # modo, payload, modifiers
    sigSelectionChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = None
        self.tool = NAV

        self.vb = EditViewBox()
        self.plot = pg.PlotWidget(viewBox=self.vb)
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Longitud", units="deg")
        self.plot.setLabel("left", "Latitud", units="deg")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plot)

        # --- capas, de abajo hacia arriba ---
        self.img_fill = pg.ImageItem()
        self.img_fill.setZValue(-100)
        self.plot.addItem(self.img_fill)

        self.it_wire = pg.PlotCurveItem(connect="pairs", pen=pg.mkPen(COL_WIRE, width=1))
        self.it_wire.setZValue(0)
        self.plot.addItem(self.it_wire)

        self.it_bnd = pg.PlotCurveItem(connect="pairs",
                                       pen=pg.mkPen(COL_BND, width=2))
        self.it_bnd.setZValue(10)
        self.plot.addItem(self.it_bnd)

        self.it_ns = {}                          # tipo -> PlotCurveItem
        for kind, col, w in (("ocean", COL_OCEAN, 3), ("land", COL_LAND, 3),
                             ("island", COL_ISLAND, 3)):
            it = pg.PlotCurveItem(connect="pairs", pen=pg.mkPen(col, width=w))
            it.setZValue(11)
            self.plot.addItem(it)
            self.it_ns[kind] = it

        self.it_hl_edges = pg.PlotCurveItem(connect="pairs",
                                            pen=pg.mkPen(COL_HL, width=2))
        self.it_hl_edges.setZValue(20)
        self.plot.addItem(self.it_hl_edges)

        self.it_hl_nodes = pg.ScatterPlotItem(size=9, pen=pg.mkPen(COL_HL, width=2),
                                              brush=pg.mkBrush(*COL_HL, 70),
                                              pxMode=True)
        self.it_hl_nodes.setZValue(21)
        self.plot.addItem(self.it_hl_nodes)

        self.it_nodes = pg.ScatterPlotItem(size=6, pen=None,
                                           brush=pg.mkBrush(*COL_NODE, 230), pxMode=True)
        self.it_nodes.setZValue(30)
        self.plot.addItem(self.it_nodes)

        self.it_sel_nodes = pg.ScatterPlotItem(size=11, pen=pg.mkPen(COL_SEL, width=2),
                                               brush=pg.mkBrush(*COL_SEL, 90),
                                               pxMode=True)
        self.it_sel_nodes.setZValue(41)
        self.plot.addItem(self.it_sel_nodes)

        self.it_sel_edges = pg.PlotCurveItem(connect="pairs",
                                             pen=pg.mkPen(COL_SEL, width=3))
        self.it_sel_edges.setZValue(40)
        self.plot.addItem(self.it_sel_edges)

        self.it_rubber = pg.PlotCurveItem(pen=pg.mkPen(COL_SEL, width=1,
                                                       style=QtCore.Qt.DashLine))
        self.it_rubber.setZValue(60)
        self.plot.addItem(self.it_rubber)

        self.it_pending = pg.ScatterPlotItem(size=13, symbol="s",
                                             pen=pg.mkPen((0, 150, 136), width=2),
                                             brush=None, pxMode=True)
        self.it_pending.setZValue(61)
        self.plot.addItem(self.it_pending)

        # --- estado ---
        self.sel_nodes = np.zeros(0, np.int64)
        self.sel_elems = np.zeros(0, np.int64)
        self.hl_nodes = np.zeros(0, np.int64)
        self.hl_elems = np.zeros(0, np.int64)
        self.pending = []                        # nodos acumulados por la herramienta
        self.show_wire = True
        self.wire_lod = True
        self.show_nodes = True
        self.show_bnd = True
        self.show_ns = True
        self.show_fill = False
        self.field_name = None
        self.field_vals = None
        self.field_kind = "node"
        self.field_range = None
        self.cmap = "viridis"
        self._drag_node = None
        self._last_render_rect = None
        self._geom_ver = -1
        self._topo_ver = -1
        self._wire_hidden = False

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(220)
        self._timer.timeout.connect(self._render_fill)
        self.vb.sigRangeChanged.connect(self._on_range)
        self.vb.sigToolClick.connect(self._on_click)
        self.vb.sigToolDrag.connect(self._on_drag)
        self.vb.sigCursorMoved.connect(self._on_cursor)

    # ---------------------------------------------------------------- estado
    def set_mesh(self, mesh, reset_view=True):
        self.mesh = mesh
        self.sel_nodes = np.zeros(0, np.int64)
        self.sel_elems = np.zeros(0, np.int64)
        self.hl_nodes = np.zeros(0, np.int64)
        self.hl_elems = np.zeros(0, np.int64)
        self.pending = []
        self._geom_ver = self._topo_ver = -1
        if mesh is not None:
            lat0 = float(np.degrees(mesh.lat0))
            self.plot.setAspectLocked(True, ratio=1.0 / max(np.cos(np.radians(lat0)), 1e-6))
        self.refresh(full=True)
        if reset_view and mesh is not None:
            self.zoom_all()

    def set_tool(self, tool):
        self.tool = tool
        self.vb.tool = tool
        self.pending = []
        self.it_pending.setData([], [])
        cur = QtCore.Qt.ArrowCursor if tool == NAV else QtCore.Qt.CrossCursor
        self.plot.setCursor(cur)
        self.sigStatus.emit(TOOL_HELP.get(tool, ""))

    def zoom_all(self):
        if self.mesh is None:
            return
        x0, y0, x1, y1 = self.mesh.bbox()
        mx, my = 0.03 * (x1 - x0 + 1e-9), 0.03 * (y1 - y0 + 1e-9)
        self.vb.setRange(xRange=(x0 - mx, x1 + mx), yRange=(y0 - my, y1 + my),
                         padding=0)

    def zoom_to_nodes(self, ids, pad=0.25):
        """Zoom sobre un conjunto de nodos, sin pasarse de cerca.

        La extension minima son ~25 aristas locales: acercarse mas dejaria la
        pantalla practicamente vacia (como pasa al centrar un solo nodo).
        """
        if self.mesh is None or len(ids) == 0:
            return
        ids = np.atleast_1d(ids)
        p = self.mesh.xy[ids]
        x0, x1 = float(p[:, 0].min()), float(p[:, 0].max())
        y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
        floor = 25.0 * self._local_edge(ids)
        dx = max(x1 - x0, floor) * (1 + pad); dy = max(y1 - y0, floor) * (1 + pad)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.vb.setRange(xRange=(cx - dx / 2, cx + dx / 2),
                         yRange=(cy - dy / 2, cy + dy / 2), padding=0)

    def _local_edge(self, ids):
        """Longitud tipica de arista (en grados) alrededor de esos nodos."""
        m = self.mesh
        xy = m.xy
        d = []
        for i in np.atleast_1d(ids)[:12]:
            nb = m.neighbors_of_node(int(i))
            if len(nb):
                d.append(np.abs(xy[nb] - xy[int(i)]).max())
        return float(np.median(d)) if d else 1e-3

    # ------------------------------------------------------------- selection
    def set_selection(self, nodes=None, elems=None):
        if nodes is not None:
            self.sel_nodes = np.unique(np.asarray(nodes, np.int64))
        if elems is not None:
            self.sel_elems = np.unique(np.asarray(elems, np.int64))
        self._draw_selection()
        self.sigSelectionChanged.emit()

    def clear_selection(self):
        self.set_selection(nodes=np.zeros(0, np.int64), elems=np.zeros(0, np.int64))

    def set_highlight(self, nodes=None, elems=None):
        self.hl_nodes = np.asarray(nodes if nodes is not None else [], np.int64)
        self.hl_elems = np.asarray(elems if elems is not None else [], np.int64)
        self._draw_highlight()

    # ------------------------------------------------------------- refresco
    def refresh(self, full=False):
        m = self.mesh
        if m is None:
            for it in (self.it_wire, self.it_bnd, self.it_hl_edges, self.it_sel_edges):
                it.setData([], [])
            self.it_nodes.setData([], [])
            self.img_fill.clear()
            return
        topo_changed = full or m.topo_version != self._topo_ver
        geom_changed = full or m.geom_version != self._geom_ver
        if topo_changed or geom_changed:
            self._draw_wire()
            self._draw_boundaries()
            self._draw_selection()
            self._draw_highlight()
            self._draw_nodes()
            self._timer.start()
        self._topo_ver = m.topo_version
        self._geom_ver = m.geom_version

    @staticmethod
    def _segments(xy, pairs):
        n = len(pairs)
        X = np.empty(n * 2); Y = np.empty(n * 2)
        X[0::2] = xy[pairs[:, 0], 0]; X[1::2] = xy[pairs[:, 1], 0]
        Y[0::2] = xy[pairs[:, 0], 1]; Y[1::2] = xy[pairs[:, 1], 1]
        return X, Y

    def _wire_too_dense(self):
        """True si la arista media mide menos de ~2 px: dibujarla seria ruido."""
        m = self.mesh
        if m is None or not self.wire_lod:
            return False
        ue, _ = m.edges()
        if len(ue) < 5000:
            return False
        (x0, x1), _ = self.vb.viewRange()
        w = max(self.plot.width(), 1)
        deg_per_px = (x1 - x0) / w
        step = max(1, len(ue) // 4000)
        d = np.abs(m.xy[ue[::step, 0], 0] - m.xy[ue[::step, 1], 0])
        return float(np.median(d)) < 2.0 * deg_per_px

    def _draw_wire(self):
        if not self.show_wire or self._wire_too_dense():
            self.it_wire.setData([], [])
            return
        ue, _ = self.mesh.edges()
        if not len(ue):
            self.it_wire.setData([], []); return
        X, Y = self._segments(self.mesh.xy, ue)
        self.it_wire.setData(X, Y)

    def _draw_boundaries(self):
        be = self.mesh.boundary_edges()
        if self.show_bnd and len(be):
            self.it_bnd.setData(*self._segments(self.mesh.xy, be))
        else:
            self.it_bnd.setData([], [])
        buckets = {"ocean": [], "land": [], "island": []}
        if self.show_ns:
            for b in self.mesh.boundaries:
                n = b.nodes
                if len(n) < 2:
                    continue
                buckets.setdefault(b.kind, []).append(
                    np.column_stack([n[:-1], n[1:]]))
        for kind, it in self.it_ns.items():
            segs = buckets.get(kind) or []
            if segs:
                it.setData(*self._segments(self.mesh.xy, np.vstack(segs)))
            else:
                it.setData([], [])

    def _draw_nodes(self):
        """Marcadores de nodo: solo cuando estan lo bastante separados en pantalla.

        El criterio es la separacion en pixeles, no el numero de nodos: asi una
        malla pequena vista completa tampoco se llena de puntos.
        """
        if not self.show_nodes or self.mesh is None:
            self.it_nodes.setData([], []); return
        (x0, x1), (y0, y1) = self.vb.viewRange()
        xy = self.mesh.xy
        live = self.mesh.live_nodes()
        p = xy[live]
        vis = live[(p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)]
        n = len(vis)
        area_px = max(self.plot.width() * self.plot.height(), 1)
        spacing = np.sqrt(area_px / n) if n else np.inf
        if n == 0 or spacing < 14.0:
            self.it_nodes.setData([], [])
        else:
            self.it_nodes.setData(xy[vis, 0], xy[vis, 1])

    def _draw_selection(self):
        m = self.mesh
        if m is None:
            return
        sn = self.sel_nodes[m.node_alive[self.sel_nodes]] if len(self.sel_nodes) else self.sel_nodes
        self.it_sel_nodes.setData(m.xy[sn, 0], m.xy[sn, 1]) if len(sn) \
            else self.it_sel_nodes.setData([], [])
        se = self.sel_elems[m.elem_alive[self.sel_elems]] if len(self.sel_elems) else self.sel_elems
        if len(se):
            t = m.tri[se]
            pairs = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
            self.it_sel_edges.setData(*self._segments(m.xy, pairs))
        else:
            self.it_sel_edges.setData([], [])

    def _draw_highlight(self):
        m = self.mesh
        if m is None:
            return
        hn = self.hl_nodes
        if len(hn):
            hn = hn[(hn >= 0) & (hn < len(m.xy))]
            self.it_hl_nodes.setData(m.xy[hn, 0], m.xy[hn, 1])
        else:
            self.it_hl_nodes.setData([], [])
        he = self.hl_elems
        if len(he):
            he = he[(he >= 0) & (he < len(m.tri))]
            t = m.tri[he]
            pairs = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
            self.it_hl_edges.setData(*self._segments(m.xy, pairs))
        else:
            self.it_hl_edges.setData([], [])

    # ------------------------------------------------------ relleno de color
    def set_field(self, name, values, kind, cmap="viridis", vrange=None):
        self.field_name = name
        self.field_vals = None if values is None else np.asarray(values, float)
        self.field_kind = kind
        self.cmap = cmap
        self.field_range = vrange
        self._last_render_rect = None
        self._timer.start(0)

    def set_show_fill(self, on):
        self.show_fill = on
        self.img_fill.setVisible(on)
        if on:
            self._last_render_rect = None
            self._timer.start(0)

    def _on_range(self):
        self._draw_nodes()
        if self.mesh is not None and self.show_wire:
            dense = self._wire_too_dense()
            if dense != self._wire_hidden:
                self._wire_hidden = dense
                self._draw_wire()
        self._timer.start()

    def _render_fill(self):
        """Rasteriza el campo activo con Agg sobre la extension visible."""
        m = self.mesh
        if m is None or not self.show_fill or self.field_vals is None:
            self.img_fill.clear(); return
        (x0, x1), (y0, y1) = self.vb.viewRange()
        mx, my = 0.20 * (x1 - x0), 0.20 * (y1 - y0)
        rect = (x0 - mx, y0 - my, x1 + mx, y1 + my)
        if self._stable(rect):
            return
        ids, t = m.live_elems()
        if not len(t):
            self.img_fill.clear(); return
        xy = m.xy
        cx = xy[t, 0]; cy = xy[t, 1]
        vis = ((cx.max(1) >= rect[0]) & (cx.min(1) <= rect[2]) &
               (cy.max(1) >= rect[1]) & (cy.min(1) <= rect[3]))
        sub = np.flatnonzero(vis)
        if not len(sub):
            self.img_fill.clear(); return
        if self.field_kind == "element" and len(self.field_vals) != len(ids):
            self.img_fill.clear(); return      # el campo quedo obsoleto tras editar
        # nunca se descartan triangulos (dejaria huecos): con muchos elementos se
        # baja la resolucion del raster, que es lo que domina el costo de Agg
        px = self._px()
        if len(sub) > 250000:
            px = (max(320, px[0] // 2), max(320, px[1] // 2))
        buf = _rasterize(xy, t[sub], self.field_vals, self.field_kind,
                         ids[sub] if self.field_kind == "element" else None,
                         rect, px, self.cmap, self.field_range, ids)
        if buf is None:
            self.img_fill.clear(); return
        self.img_fill.setImage(buf[::-1], autoLevels=False)
        self.img_fill.setRect(QtCore.QRectF(rect[0], rect[1],
                                            rect[2] - rect[0], rect[3] - rect[1]))
        self._last_render_rect = rect

    def _px(self):
        s = self.plot.size()
        return max(320, min(s.width(), 1600)), max(320, min(s.height(), 1600))

    def _stable(self, rect):
        """True si la nueva extension esta contenida en la ya rasterizada."""
        r = self._last_render_rect
        if r is None:
            return False
        w, h = r[2] - r[0], r[3] - r[1]
        nw, nh = rect[2] - rect[0], rect[3] - rect[1]
        if not (0.7 < nw / w < 1.3):
            return False
        return (rect[0] >= r[0] - 1e-12 and rect[1] >= r[1] - 1e-12 and
                rect[2] <= r[2] + 1e-12 and rect[3] <= r[3] + 1e-12)

    # -------------------------------------------------------------- eventos
    def _on_cursor(self, p):
        self.sigCursor.emit(p[0], p[1])

    def _on_click(self, p, mods):
        if self.mesh is None:
            return
        self.sigToolAction.emit(self.tool, ("click", p), mods)

    def _on_drag(self, start, cur, state):
        st, mods = state
        if self.mesh is None:
            return
        if self.tool in (SEL_NODE, SEL_ELEM):
            if st == "finish":
                self.it_rubber.setData([], [])
                self.sigToolAction.emit(self.tool, ("box", start, cur), mods)
            else:
                x0, y0 = start; x1, y1 = cur
                self.it_rubber.setData([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
        elif self.tool == MOVE_NODE:
            self.sigToolAction.emit(self.tool, ("drag", start, cur, st), mods)

    def set_pending(self, ids):
        self.pending = list(ids)
        if self.pending and self.mesh is not None:
            p = self.mesh.xy[np.asarray(self.pending, np.int64)]
            self.it_pending.setData(p[:, 0], p[:, 1])
        else:
            self.it_pending.setData([], [])


# --------------------------------------------------------- rasterizador Agg
_FIG = None


def _rasterize(xy, tri, vals, kind, elem_ids, rect, px, cmap, vrange, all_ids):
    """Devuelve un buffer RGBA (H, W, 4) con el campo dibujado en `rect`."""
    global _FIG
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.tri as mtri

    w, h = px
    dpi = 100.0
    if _FIG is None or tuple(_FIG.get_size_inches()) != (w / dpi, h / dpi):
        _FIG = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        FigureCanvasAgg(_FIG)
    fig = _FIG
    fig.clf()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(rect[0], rect[2]); ax.set_ylim(rect[1], rect[3])
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)

    try:
        trg = mtri.Triangulation(xy[:, 0], xy[:, 1], tri)
        kw = dict(cmap=cmap)
        if vrange is not None:
            kw["vmin"], kw["vmax"] = vrange
        if kind == "node":
            ax.tripcolor(trg, vals, shading="gouraud", **kw)
        else:
            pos = np.searchsorted(all_ids, elem_ids)
            ax.tripcolor(trg, facecolors=vals[pos], **kw)
    except Exception:
        return None
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()

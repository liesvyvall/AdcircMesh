"""Paneles acoplables: control de calidad, estadisticas, propiedades, capas y
fronteras."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.quality import ERROR, WARN, INFO, OK, NODE, ELEM
from ..core.mesh import OCEAN, LAND, ISLAND, IBTYPE_LABELS

SEV_COLOR = {ERROR: "#c81e37", WARN: "#b45309", INFO: "#1d4ed8", OK: "#15803d"}
SEV_ICON = {ERROR: "✘", WARN: "⚠", INFO: "ℹ", OK: "✔"}


def plural(n, uno, varios):
    return f"{n:,} {uno if n == 1 else varios}"


# ---------------------------------------------------------------- QC
class QCPanel(QtWidgets.QWidget):
    sigHighlight = QtCore.Signal(str, object)     # kind, ids
    sigZoom = QtCore.Signal(str, object)          # kind, ids
    sigRun = QtCore.Signal(dict)                  # parametros
    sigSelect = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.report = None
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)

        # parametros
        form = QtWidgets.QGridLayout()
        self.sp_dt = QtWidgets.QDoubleSpinBox(); self.sp_dt.setRange(0.01, 600); self.sp_dt.setValue(1.0)
        self.sp_dt.setSuffix(" s"); self.sp_dt.setDecimals(2)
        self.sp_ang = QtWidgets.QDoubleSpinBox(); self.sp_ang.setRange(1, 60); self.sp_ang.setValue(30.0)
        self.sp_ang.setSuffix(" deg")
        self.sp_val = QtWidgets.QSpinBox(); self.sp_val.setRange(4, 30); self.sp_val.setValue(10)
        self.sp_dep = QtWidgets.QDoubleSpinBox(); self.sp_dep.setRange(0, 100); self.sp_dep.setValue(1.0)
        self.sp_dep.setSuffix(" m")
        self.sp_grad = QtWidgets.QDoubleSpinBox(); self.sp_grad.setRange(1.05, 5.0)
        self.sp_grad.setValue(1.5); self.sp_grad.setSingleStep(0.05)
        for r, (lab, w) in enumerate([("Paso de tiempo objetivo", self.sp_dt),
                                      ("Angulo minimo aceptable", self.sp_ang),
                                      ("Valencia maxima", self.sp_val),
                                      ("Profundidad minima", self.sp_dep),
                                      ("Gradacion maxima", self.sp_grad)]):
            form.addWidget(QtWidgets.QLabel(lab), r, 0)
            form.addWidget(w, r, 1)
        v.addLayout(form)

        h = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Ejecutar control de calidad")
        self.btn_run.clicked.connect(self._emit_run)
        self.btn_export = QtWidgets.QPushButton("Exportar…")
        h.addWidget(self.btn_run, 3); h.addWidget(self.btn_export, 1)
        v.addLayout(h)

        self.summary = QtWidgets.QLabel("Sin resultados")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Chequeo", "Cantidad"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setWordWrap(True)
        self.tree.setIndentation(12)
        self.tree.setTextElideMode(QtCore.Qt.ElideRight)
        hh = self.tree.header()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tree.itemSelectionChanged.connect(self._on_sel)
        self.tree.itemDoubleClicked.connect(self._on_dbl)
        v.addWidget(self.tree, 1)

        hb = QtWidgets.QHBoxLayout()
        self.chk_only_bad = QtWidgets.QCheckBox("Solo problemas")
        self.chk_only_bad.setChecked(True)
        self.chk_only_bad.toggled.connect(lambda _: self.set_report(self.report))
        self.lb_case = QtWidgets.QLabel("")
        self.lb_case.setStyleSheet("color:#6b7280")
        b_prev = QtWidgets.QToolButton(); b_prev.setText("◀")
        b_prev.setToolTip("Caso anterior")
        b_prev.clicked.connect(lambda: self._step(-1))
        b_next = QtWidgets.QToolButton(); b_next.setText("▶")
        b_next.setToolTip("Caso siguiente")
        b_next.clicked.connect(lambda: self._step(+1))
        b_sel = QtWidgets.QPushButton("Seleccionar")
        b_sel.setToolTip("Pasa los elementos/nodos del chequeo a la seleccion activa")
        b_sel.clicked.connect(self._push_selection)
        hb.addWidget(self.chk_only_bad); hb.addStretch(1)
        hb.addWidget(self.lb_case); hb.addWidget(b_prev); hb.addWidget(b_next)
        hb.addWidget(b_sel)
        v.addLayout(hb)
        self._case = 0

    def params(self):
        return dict(dt_target=self.sp_dt.value(), min_angle=self.sp_ang.value(),
                    max_valence=self.sp_val.value(), min_depth=self.sp_dep.value(),
                    gradation=self.sp_grad.value())

    def _emit_run(self):
        self.sigRun.emit(self.params())

    def set_report(self, rep):
        self.report = rep
        self.tree.clear()
        if rep is None:
            self.summary.setText("Sin resultados")
            return
        groups = {}
        for c in rep.checks:
            if self.chk_only_bad.isChecked() and c.severity == OK:
                continue
            g = groups.get(c.group)
            if g is None:
                g = QtWidgets.QTreeWidgetItem(self.tree, [c.group, ""])
                f = g.font(0); f.setBold(True); g.setFont(0, f)
                g.setExpanded(True)
                groups[c.group] = g
            it = QtWidgets.QTreeWidgetItem(g, [f"{SEV_ICON[c.severity]}  {c.label}",
                                               f"{c.count:,}"])
            it.setForeground(0, QtGui.QColor(SEV_COLOR[c.severity]))
            it.setTextAlignment(1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            tip = c.label + (f"<br><i>{c.detail}</i>" if c.detail else "")
            it.setToolTip(0, tip)
            it.setData(0, QtCore.Qt.UserRole, c)
        ne, nw = rep.n_errors, rep.n_warnings
        color = SEV_COLOR[ERROR] if ne else (SEV_COLOR[WARN] if nw else SEV_COLOR[OK])
        txt = (f"<b style='color:{color}'>{plural(ne, 'error', 'errores')} "
               f"&middot; {plural(nw, 'aviso', 'avisos')}</b>"
               "<br><span style='color:#6b7280'>Doble clic en un chequeo para "
               "hacer zoom sobre los casos</span>")
        self.summary.setText(txt)

    def _current(self):
        its = self.tree.selectedItems()
        if not its:
            return None
        return its[0].data(0, QtCore.Qt.UserRole)

    def _on_sel(self):
        self._case = 0
        c = self._current()
        if c is None or c.kind not in (NODE, ELEM) or not len(c.ids):
            self.lb_case.setText("")
            self.sigHighlight.emit("none", np.zeros(0, np.int64))
            return
        self.lb_case.setText(f"caso 1/{len(c.ids):,}")
        self.sigHighlight.emit(c.kind, c.ids)

    def _on_dbl(self, item, col):
        c = item.data(0, QtCore.Qt.UserRole)
        if c is not None and c.kind in (NODE, ELEM) and len(c.ids):
            self.sigZoom.emit(c.kind, c.ids)

    def _step(self, d):
        """Recorre los casos del chequeo activo uno por uno, como en SMS."""
        c = self._current()
        if c is None or c.kind not in (NODE, ELEM) or not len(c.ids):
            return
        self._case = (self._case + d) % len(c.ids)
        self.lb_case.setText(f"caso {self._case + 1:,}/{len(c.ids):,}")
        self.sigZoom.emit(c.kind, c.ids[self._case:self._case + 1])

    def _push_selection(self):
        c = self._current()
        if c is not None and c.kind in (NODE, ELEM) and len(c.ids):
            self.sigSelect.emit(c.kind, c.ids)


# ------------------------------------------------------------- estadisticas
class StatsPanel(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 6, parent)
        self.setHorizontalHeaderLabels(["Metrica", "min", "p1", "mediana", "p99", "max"])
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

    def set_report(self, rep):
        self.setRowCount(0)
        if rep is None:
            return
        for name, s in rep.stats.items():
            r = self.rowCount(); self.insertRow(r)
            unit = f" [{s['unit']}]" if s["unit"] else ""
            self.setItem(r, 0, QtWidgets.QTableWidgetItem(name + unit))
            for c, k in enumerate(("min", "p1", "med", "p99", "max"), 1):
                v = s[k]
                txt = f"{v:,.4g}" if abs(v) < 1e5 else f"{v:,.4e}"
                it = QtWidgets.QTableWidgetItem(txt)
                it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.setItem(r, c, it)


# --------------------------------------------------------------- propiedades
class PropsPanel(QtWidgets.QWidget):
    sigApplyDepth = QtCore.Signal(object, float)
    sigApplyXY = QtCore.Signal(int, float, float)
    sigZoom = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = None
        self._nodes = np.zeros(0, np.int64)
        self._elems = np.zeros(0, np.int64)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        self.info = QtWidgets.QTextEdit(); self.info.setReadOnly(True)
        self.info.setMinimumHeight(150)
        v.addWidget(self.info, 1)

        g = QtWidgets.QGroupBox("Editar seleccion")
        f = QtWidgets.QGridLayout(g)
        self.sp_z = QtWidgets.QDoubleSpinBox(); self.sp_z.setRange(-1e5, 1e5)
        self.sp_z.setDecimals(3); self.sp_z.setSuffix(" m")
        b_z = QtWidgets.QPushButton("Aplicar profundidad")
        b_z.clicked.connect(lambda: self.sigApplyDepth.emit(self._nodes, self.sp_z.value()))
        self.sp_x = QtWidgets.QDoubleSpinBox(); self.sp_x.setRange(-360, 360); self.sp_x.setDecimals(8)
        self.sp_y = QtWidgets.QDoubleSpinBox(); self.sp_y.setRange(-90, 90); self.sp_y.setDecimals(8)
        b_xy = QtWidgets.QPushButton("Aplicar coordenadas")
        b_xy.clicked.connect(self._apply_xy)
        f.addWidget(QtWidgets.QLabel("Profundidad"), 0, 0); f.addWidget(self.sp_z, 0, 1)
        f.addWidget(b_z, 0, 2)
        f.addWidget(QtWidgets.QLabel("Longitud"), 1, 0); f.addWidget(self.sp_x, 1, 1)
        f.addWidget(QtWidgets.QLabel("Latitud"), 2, 0); f.addWidget(self.sp_y, 2, 1)
        f.addWidget(b_xy, 1, 2, 2, 1)
        v.addWidget(g)

    def _apply_xy(self):
        if len(self._nodes) == 1:
            self.sigApplyXY.emit(int(self._nodes[0]), self.sp_x.value(), self.sp_y.value())

    def update_view(self, mesh, nodes, elems, report=None):
        self.mesh = mesh
        self._nodes = np.asarray(nodes, np.int64)
        self._elems = np.asarray(elems, np.int64)
        if mesh is None:
            self.info.setHtml(
                "<span style='color:#6b7280'>Abre una malla y selecciona nodos o "
                "elementos para ver aqui sus propiedades.</span>")
            return
        L = []
        if len(self._nodes) == 1:
            i = int(self._nodes[0])
            x, y = mesh.xy[i]
            val = mesh.valence()[i]
            onb = mesh.boundary_node_mask()[i]
            self.sp_z.setValue(float(mesh.z[i]))
            self.sp_x.setValue(float(x)); self.sp_y.setValue(float(y))
            L.append(f"<b>Nodo {i + 1}</b> (indice interno {i})")
            L.append(f"lon {x:.7f}&deg; &nbsp; lat {y:.7f}&deg;")
            L.append(f"profundidad <b>{mesh.z[i]:.3f} m</b>")
            L.append(f"valencia {val} &nbsp;&middot;&nbsp; "
                     f"{'en la frontera' if onb else 'interior'}")
            els = mesh.elems_of_node(i)
            L.append(f"elementos incidentes: {len(els)}")
            bl = [b.label for b in mesh.boundaries if i in set(b.nodes.tolist())]
            if bl:
                L.append("fronteras: " + ", ".join(bl))
        elif len(self._nodes) > 1:
            z = mesh.z[self._nodes]
            L.append(f"<b>{len(self._nodes):,} nodos seleccionados</b>")
            L.append(f"profundidad: min {z.min():.2f} &nbsp; media {z.mean():.2f} "
                     f"&nbsp; max {z.max():.2f} m")
            self.sp_z.setValue(float(z.mean()))
        if len(self._elems) == 1:
            k = int(self._elems[0])
            tri = mesh.tri[k]
            from ..core.quality import signed_area, edge_lengths, angles_deg, shape_quality
            t1 = tri.reshape(1, 3)
            A = signed_area(mesh.pm, t1); Lg = edge_lengths(mesh.pm, t1)
            ang = angles_deg(Lg); q = shape_quality(A, Lg)
            L.append(f"<br><b>Elemento {k + 1}</b> (indice interno {k})")
            L.append(f"nodos: {', '.join(str(v + 1) for v in tri)}")
            L.append(f"area {abs(A[0]):,.1f} m&sup2; &nbsp; orientacion "
                     f"{'CCW' if A[0] > 0 else '<span style=\"color:#ff5468\">CW</span>'}")
            L.append(f"aristas: {Lg[0, 0]:,.1f} / {Lg[0, 1]:,.1f} / {Lg[0, 2]:,.1f} m")
            L.append(f"angulos: {ang[0, 0]:.1f} / {ang[0, 1]:.1f} / {ang[0, 2]:.1f} deg")
            L.append(f"calidad de forma <b>{q[0]:.3f}</b>")
            zc = mesh.z[tri].mean()
            from ..core.mesh import G
            dt = Lg[0].min() / np.sqrt(G * max(zc, 0.1))
            L.append(f"profundidad media {zc:.2f} m &nbsp; dt maximo (CFL) "
                     f"<b>{dt:.3f} s</b>")
        elif len(self._elems) > 1:
            L.append(f"<br><b>{len(self._elems):,} elementos seleccionados</b>")
        if not L:
            L = ["<span style='color:#6b7280'>Nada seleccionado.<br>"
                 "Usa las herramientas de seleccion de la barra.</span>"]
        self.info.setHtml("<br>".join(L))


# --------------------------------------------------------------------- capas
class LayersPanel(QtWidgets.QWidget):
    sigLayers = QtCore.Signal(dict)
    sigField = QtCore.Signal(object)     # (name, cmap, vmin, vmax) o None

    CMAPS = ["viridis", "turbo", "RdYlGn", "RdYlGn_r", "Spectral_r", "terrain",
             "cividis", "magma", "coolwarm"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.report = None
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        g = QtWidgets.QGroupBox("Capas visibles")
        gv = QtWidgets.QVBoxLayout(g)
        self.chks = {}
        for key, lab, on in (("wire", "Aristas de la malla", True),
                             ("nodes", "Nodos (con zoom cercano)", True),
                             ("bnd", "Borde de la malla", True),
                             ("ns", "Nodestrings de frontera", True),
                             ("fill", "Relleno de color", False),
                             ("lod", "Ocultar aristas al alejarse", True)):
            c = QtWidgets.QCheckBox(lab); c.setChecked(on)
            c.toggled.connect(self._emit)
            gv.addWidget(c); self.chks[key] = c
        v.addWidget(g)

        g2 = QtWidgets.QGroupBox("Campo de color")
        f = QtWidgets.QGridLayout(g2)
        self.cb_field = QtWidgets.QComboBox()
        self.cb_cmap = QtWidgets.QComboBox(); self.cb_cmap.addItems(self.CMAPS)
        self.sp_min = QtWidgets.QDoubleSpinBox(); self.sp_min.setRange(-1e9, 1e9); self.sp_min.setDecimals(4)
        self.sp_max = QtWidgets.QDoubleSpinBox(); self.sp_max.setRange(-1e9, 1e9); self.sp_max.setDecimals(4)
        self.chk_auto = QtWidgets.QCheckBox("Rango automatico (p1-p99)"); self.chk_auto.setChecked(True)
        f.addWidget(QtWidgets.QLabel("Campo"), 0, 0); f.addWidget(self.cb_field, 0, 1)
        f.addWidget(QtWidgets.QLabel("Paleta"), 1, 0); f.addWidget(self.cb_cmap, 1, 1)
        f.addWidget(self.chk_auto, 2, 0, 1, 2)
        f.addWidget(QtWidgets.QLabel("min"), 3, 0); f.addWidget(self.sp_min, 3, 1)
        f.addWidget(QtWidgets.QLabel("max"), 4, 0); f.addWidget(self.sp_max, 4, 1)
        self.bar = _ColorBar()
        f.addWidget(self.bar, 5, 0, 1, 2)
        v.addWidget(g2)
        v.addStretch(1)

        for w in (self.cb_field, self.cb_cmap):
            w.currentIndexChanged.connect(self._field_changed)
        self.chk_auto.toggled.connect(self._field_changed)
        for w in (self.sp_min, self.sp_max):
            w.editingFinished.connect(self._field_changed)

    def _emit(self):
        self.sigLayers.emit({k: c.isChecked() for k, c in self.chks.items()})

    def set_report(self, rep):
        self.report = rep
        cur = self.cb_field.currentText()
        self.cb_field.blockSignals(True)
        self.cb_field.clear()
        if rep is not None:
            self.cb_field.addItems(list(rep.fields))
            i = self.cb_field.findText(cur)
            self.cb_field.setCurrentIndex(i if i >= 0 else 0)
        self.cb_field.blockSignals(False)
        self._field_changed()

    def _field_changed(self):
        if self.report is None or not self.cb_field.count():
            self.sigField.emit(None); return
        name = self.cb_field.currentText()
        vals, _ = self.report.fields[name]
        if self.chk_auto.isChecked():
            vmin = float(np.nanpercentile(vals, 1)); vmax = float(np.nanpercentile(vals, 99))
            self.sp_min.blockSignals(True); self.sp_max.blockSignals(True)
            self.sp_min.setValue(vmin); self.sp_max.setValue(vmax)
            self.sp_min.blockSignals(False); self.sp_max.blockSignals(False)
        vmin, vmax = self.sp_min.value(), self.sp_max.value()
        if vmax <= vmin:
            vmax = vmin + 1e-9
        cmap = self.cb_cmap.currentText()
        self.bar.configure(cmap, vmin, vmax, name)
        self.sigField.emit((name, cmap, vmin, vmax))


class _ColorBar(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(46)
        self._grad = None
        self._txt = ("", "", "")

    def configure(self, cmap, vmin, vmax, name):
        import matplotlib.cm as cm
        c = cm.get_cmap(cmap) if hasattr(cm, "get_cmap") else None
        if c is None:
            import matplotlib
            c = matplotlib.colormaps[cmap]
        g = QtGui.QLinearGradient(0, 0, 1, 0)
        g.setCoordinateMode(QtGui.QGradient.ObjectBoundingMode)
        for s in np.linspace(0, 1, 16):
            r, gg, b, _ = c(float(s))
            g.setColorAt(float(s), QtGui.QColor.fromRgbF(r, gg, b))
        self._grad = g
        self._txt = (f"{vmin:,.4g}", name, f"{vmax:,.4g}")
        self.update()

    def paintEvent(self, ev):
        if self._grad is None:
            return
        p = QtGui.QPainter(self)
        r = self.rect().adjusted(2, 2, -2, -20)
        p.fillRect(r, QtGui.QBrush(self._grad))
        p.setPen(QtGui.QColor("#9aa2b1"))
        p.drawRect(r)
        fm = p.fontMetrics()
        y = self.rect().bottom() - 4
        p.setPen(QtGui.QColor("#1f2430"))
        p.drawText(2, y, self._txt[0])
        p.drawText(self.rect().right() - 2 - fm.horizontalAdvance(self._txt[2]), y, self._txt[2])
        p.drawText((self.width() - fm.horizontalAdvance(self._txt[1])) // 2, y, self._txt[1])


# ---------------------------------------------------------------- fronteras
class BoundaryPanel(QtWidgets.QWidget):
    sigHighlight = QtCore.Signal(object)
    sigZoom = QtCore.Signal(object)
    sigChanged = QtCore.Signal()
    sigAuto = QtCore.Signal()
    sigFromSelection = QtCore.Signal(str, int)
    sigDelete = QtCore.Signal(object)
    sigSetType = QtCore.Signal(int, str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = None
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        self.list = QtWidgets.QTreeWidget()
        self.list.setHeaderLabels(["Frontera", "Tipo", "IBTYPE", "Nodos"])
        self.list.setAlternatingRowColors(True)
        self.list.itemSelectionChanged.connect(self._on_sel)
        self.list.itemDoubleClicked.connect(self._on_dbl)
        v.addWidget(self.list, 1)

        h = QtWidgets.QHBoxLayout()
        b_auto = QtWidgets.QPushButton("Generar automaticamente")
        b_auto.clicked.connect(self.sigAuto.emit)
        b_del = QtWidgets.QPushButton("Borrar")
        b_del.clicked.connect(self._delete)
        h.addWidget(b_auto, 2); h.addWidget(b_del, 1)
        v.addLayout(h)

        g = QtWidgets.QGroupBox("Crear desde la seleccion de nodos")
        f = QtWidgets.QGridLayout(g)
        self.cb_kind = QtWidgets.QComboBox()
        self.cb_kind.addItems(["Frontera abierta (oceano)", "Tierra", "Isla"])
        self.cb_ibt = QtWidgets.QComboBox()
        self.cb_ibt.addItems([IBTYPE_LABELS[k] for k in sorted(IBTYPE_LABELS)])
        self.cb_ibt.setCurrentText(IBTYPE_LABELS[20])
        b_new = QtWidgets.QPushButton("Crear nodestring")
        b_new.clicked.connect(self._create)
        b_set = QtWidgets.QPushButton("Cambiar tipo de la seleccionada")
        b_set.clicked.connect(self._set_type)
        f.addWidget(QtWidgets.QLabel("Tipo"), 0, 0); f.addWidget(self.cb_kind, 0, 1)
        f.addWidget(QtWidgets.QLabel("IBTYPE"), 1, 0); f.addWidget(self.cb_ibt, 1, 1)
        f.addWidget(b_new, 2, 0, 1, 2)
        f.addWidget(b_set, 3, 0, 1, 2)
        v.addWidget(g)

    def _kind(self):
        return [OCEAN, LAND, ISLAND][self.cb_kind.currentIndex()]

    def _ibtype(self):
        return int(self.cb_ibt.currentText().split()[0])

    def set_mesh(self, mesh):
        self.mesh = mesh
        self.list.clear()
        if mesh is None:
            return
        names = {OCEAN: "Oceano", LAND: "Tierra", ISLAND: "Isla"}
        for i, b in enumerate(mesh.boundaries):
            it = QtWidgets.QTreeWidgetItem(
                self.list, [b.name or f"#{i + 1}", names.get(b.kind, b.kind),
                            "" if b.ibtype is None else str(b.ibtype), f"{len(b):,}"])
            it.setData(0, QtCore.Qt.UserRole, i)
            col = {OCEAN: "#007acc", LAND: "#e06c10", ISLAND: "#118a44"}.get(b.kind)
            if col:
                it.setForeground(1, QtGui.QColor(col))
        for c in range(4):
            self.list.resizeColumnToContents(c)

    def _indices(self):
        return [it.data(0, QtCore.Qt.UserRole) for it in self.list.selectedItems()]

    def _on_sel(self):
        idx = self._indices()
        if not idx or self.mesh is None:
            self.sigHighlight.emit(np.zeros(0, np.int64)); return
        ids = np.concatenate([self.mesh.boundaries[i].nodes for i in idx])
        self.sigHighlight.emit(ids)

    def _on_dbl(self, item, col):
        i = item.data(0, QtCore.Qt.UserRole)
        if self.mesh is not None and i is not None:
            self.sigZoom.emit(self.mesh.boundaries[i].nodes)

    def _delete(self):
        idx = self._indices()
        if idx:
            self.sigDelete.emit(idx)

    def _create(self):
        self.sigFromSelection.emit(self._kind(), self._ibtype())

    def _set_type(self):
        idx = self._indices()
        if idx:
            self.sigSetType.emit(idx[0], self._kind(), self._ibtype())


# -------------------------------------------------------------------- log
class LogPanel(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(4000)
        f = QtGui.QFont("Menlo"); f.setStyleHint(QtGui.QFont.Monospace); f.setPointSize(12)
        self.setFont(f)

    def log(self, msg):
        for ln in str(msg).splitlines():
            self.appendPlainText(ln)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

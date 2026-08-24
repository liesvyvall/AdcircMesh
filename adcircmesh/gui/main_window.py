"""Ventana principal de AdcircMesh."""
from __future__ import annotations

import os
import time
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import edit, repair
from ..core.mesh import Mesh, Boundary, OCEAN, LAND, ISLAND
from ..core.meshio import load_mesh, save_mesh
from ..core.quality import run_qc, NODE, ELEM
from . import render
from .commands import push
from .panels import (QCPanel, StatsPanel, PropsPanel, LayersPanel,
                     BoundaryPanel, LogPanel)

APP_NAME = "AdcircMesh"
SUBTITLE = "Editor de mallas no estructuradas ADCIRC / SWAN"

STYLE = """
* { font-size: 13px; }
QMainWindow, QWidget { background: #f4f5f7; color: #1f2430; }
QDockWidget { font-size: 15px; font-weight: 600; }
QDockWidget::title { background: #e7eaef; padding: 8px 12px;
    border-bottom: 1px solid #d0d5dd; font-size: 15px; }
QGroupBox { border: 1px solid #d0d5dd; border-radius: 6px; margin-top: 18px;
    padding-top: 10px; background: #ffffff; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: #4b5563; font-size: 14px; font-weight: 600; }
QTreeWidget, QTableWidget, QPlainTextEdit, QTextEdit {
    background: #ffffff; border: 1px solid #d0d5dd; border-radius: 4px;
    alternate-background-color: #f7f8fa; selection-background-color: #d6e4ff;
    selection-color: #1f2430; }
QHeaderView::section { background: #eef0f4; border: 0; border-right: 1px solid #d0d5dd;
    border-bottom: 1px solid #d0d5dd; padding: 6px; color: #4b5563;
    font-size: 13px; font-weight: 600; }
QPushButton { background: #ffffff; border: 1px solid #c3c9d4; border-radius: 5px;
    padding: 6px 12px; }
QPushButton:hover { background: #eef2f8; border-color: #9fb3d1; }
QPushButton:pressed { background: #dfe6f2; }
QToolBar { background: #e7eaef; border-bottom: 1px solid #d0d5dd; spacing: 2px; padding: 4px; }
QToolButton { padding: 6px 9px; border-radius: 5px; font-size: 13px; }
QToolButton:hover { background: #d8dee8; }
QToolButton:checked { background: #2563eb; color: #ffffff; }
QMenuBar { background: #e7eaef; font-size: 14px; }
QMenuBar::item { padding: 5px 10px; }
QMenuBar::item:selected { background: #2563eb; color: #ffffff; }
QMenu { background: #ffffff; border: 1px solid #d0d5dd; }
QMenu::item { padding: 5px 24px 5px 20px; }
QMenu::item:selected { background: #2563eb; color: #ffffff; }
QStatusBar { background: #e7eaef; border-top: 1px solid #d0d5dd; }
QStatusBar QLabel { font-size: 13px; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #ffffff; border: 1px solid #c3c9d4; border-radius: 4px; padding: 4px 6px; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border-color: #2563eb; }
QCheckBox::indicator { width: 15px; height: 15px; }
QTabBar::tab { background: #e7eaef; border: 1px solid #d0d5dd; border-bottom: 0;
    padding: 6px 14px; font-size: 13px; }
QTabBar::tab:selected { background: #ffffff; color: #2563eb; font-weight: 600; }
QProgressBar { border: 1px solid #c3c9d4; border-radius: 4px; text-align: center;
    background: #ffffff; }
QProgressBar::chunk { background: #2563eb; }
QScrollBar:vertical, QScrollBar:horizontal { background: #f0f2f5; border: 0; }
QScrollBar::handle { background: #c3c9d4; border-radius: 5px; min-height: 24px; }
QScrollBar::handle:hover { background: #a8b0bf; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
"""

# (modo, etiqueta de menu, glifo, atajo, etiqueta corta para la barra)
TOOLS = [
    (render.NAV,         "Navegar",           "✥", "Escape", "Navegar"),
    (render.SEL_NODE,    "Selec. nodos",      "◦", "N", "Nodos"),
    (render.SEL_ELEM,    "Selec. elementos",  "△", "E", "Elementos"),
    (render.MOVE_NODE,   "Mover nodo",        "✛", "M", "Mover"),
    (render.ADD_NODE,    "Anadir nodo",       "＋", "A", "Anadir"),
    (render.DEL_NODE,    "Borrar nodo",       "－", "D", "Borrar nodo"),
    (render.DEL_ELEM,    "Borrar elemento",   "⌫", "X", "Borrar elem."),
    (render.SPLIT_EDGE,  "Dividir arista",    "⊣", "S", "Dividir"),
    (render.SWAP_EDGE,   "Voltear arista",    "⇄", "F", "Voltear"),
    (render.CREATE_ELEM, "Crear elemento",    "▲", "C", "Crear"),
    (render.MERGE_NODES, "Fusionar nodos",    "⊕", "G", "Fusionar"),
]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.mesh = None
        self.path = None
        self.report = None
        self.stack = QtGui.QUndoStack(self)
        self.stack.setUndoLimit(200)
        self._pending = []
        self._drag_node = None
        self._drag_from = None

        self.setWindowTitle(f"{APP_NAME} — {SUBTITLE}")
        self.resize(1680, 1000)
        self.setStyleSheet(STYLE)

        self.view = render.MeshView()
        self.setCentralWidget(self.view)
        self.view.sigCursor.connect(self._on_cursor)
        self.view.sigToolAction.connect(self._on_tool_action)
        self.view.sigSelectionChanged.connect(self._on_selection)

        self._build_docks()
        self._build_actions()
        self._build_status()
        self._set_tool(render.NAV)
        self.props.update_view(None, [], [])
        self._update_title()
        self.log(f"{APP_NAME} listo. Abre un fort.14 o un .2dm para empezar.")

    # ------------------------------------------------------------------ docks
    def _build_docks(self):
        def dock(title, w, area, name):
            d = QtWidgets.QDockWidget(title, self)
            d.setObjectName(name)
            d.setWidget(w)
            self.addDockWidget(area, d)
            return d

        self.qc = QCPanel(); self.qc.sigRun.connect(self.run_qc)
        self.qc.sigHighlight.connect(self._hl_from_qc)
        self.qc.sigZoom.connect(self._zoom_check)
        self.qc.sigSelect.connect(self._select_from_qc)
        self.qc.btn_export.clicked.connect(self.export_report)
        self.d_qc = dock("Control de calidad", self.qc,
                         QtCore.Qt.RightDockWidgetArea, "dock_qc")

        self.stats = StatsPanel()
        self.d_stats = dock("Estadisticas", self.stats,
                            QtCore.Qt.RightDockWidgetArea, "dock_stats")

        self.props = PropsPanel()
        self.props.sigApplyDepth.connect(self._apply_depth)
        self.props.sigApplyXY.connect(self._apply_xy)
        self.d_props = dock("Propiedades", self.props,
                            QtCore.Qt.LeftDockWidgetArea, "dock_props")

        self.layers = LayersPanel()
        self.layers.sigLayers.connect(self._apply_layers)
        self.layers.sigField.connect(self._apply_field)
        self.d_layers = dock("Capas y color", self.layers,
                             QtCore.Qt.LeftDockWidgetArea, "dock_layers")

        self.bnd = BoundaryPanel()
        self.bnd.sigHighlight.connect(lambda ids: self.view.set_highlight(nodes=ids))
        self.bnd.sigZoom.connect(self._zoom_nodes)
        self.bnd.sigAuto.connect(self.auto_boundaries)
        self.bnd.sigFromSelection.connect(self._boundary_from_selection)
        self.bnd.sigDelete.connect(self._delete_boundaries)
        self.bnd.sigSetType.connect(self._set_boundary_type)
        self.d_bnd = dock("Fronteras (nodestrings)", self.bnd,
                          QtCore.Qt.LeftDockWidgetArea, "dock_bnd")

        self.logpanel = LogPanel()
        self.d_log = dock("Registro", self.logpanel,
                          QtCore.Qt.BottomDockWidgetArea, "dock_log")
        self.d_log.setMaximumHeight(220)

        self.tabifyDockWidget(self.d_qc, self.d_stats)
        self.d_qc.raise_()
        self.tabifyDockWidget(self.d_props, self.d_layers)
        self.tabifyDockWidget(self.d_layers, self.d_bnd)
        self.d_props.raise_()
        self.resizeDocks([self.d_props, self.d_qc], [320, 480], QtCore.Qt.Horizontal)
        self.resizeDocks([self.d_log], [170], QtCore.Qt.Vertical)

    # --------------------------------------------------------------- acciones
    def _build_actions(self):
        mb = self.menuBar()
        tb = self.addToolBar("Principal")
        tb.setObjectName("tb_main")
        tb.setIconSize(QtCore.QSize(20, 20))

        def act(text, slot, short=None, tip=None, checkable=False):
            a = QtGui.QAction(text, self)
            if short:
                a.setShortcut(short)
            a.setToolTip(tip or text)
            a.setCheckable(checkable)
            if slot:
                a.triggered.connect(slot)
            return a

        # --- Archivo
        m = mb.addMenu("&Archivo")
        self.a_open = act("Abrir malla…", self.open_file, "Ctrl+O")
        self.a_open.setIconText("Abrir")
        self.a_save = act("Guardar", self.save_file, "Ctrl+S")
        self.a_saveas = act("Guardar como…", self.save_file_as, "Ctrl+Shift+S")
        self.a_export = act("Exportar reporte de calidad…", self.export_report)
        m.addActions([self.a_open, self.a_save, self.a_saveas])
        m.addSeparator(); m.addAction(self.a_export)
        m.addSeparator(); m.addAction(act("Salir", self.close, "Ctrl+Q"))
        tb.addAction(self.a_open); tb.addAction(self.a_save); tb.addSeparator()

        # --- Editar
        m = mb.addMenu("&Editar")
        self.a_undo = self.stack.createUndoAction(self, "Deshacer")
        self.a_undo.setShortcut("Ctrl+Z")
        self.a_undo.setIconText("Deshacer")
        self.a_redo = self.stack.createRedoAction(self, "Rehacer")
        self.a_redo.setShortcut("Ctrl+Shift+Z")
        self.a_redo.setIconText("Rehacer")
        m.addAction(self.a_undo); m.addAction(self.a_redo)
        m.addSeparator()
        m.addAction(act("Seleccionar todo (nodos)", self._select_all_nodes, "Ctrl+A"))
        m.addAction(act("Limpiar seleccion", self._clear_sel, "Ctrl+D"))
        m.addAction(act("Invertir seleccion", self._invert_sel))
        m.addSeparator()
        m.addAction(act("Borrar seleccion", self._delete_selection, "Del"))
        m.addAction(act("Suavizar seleccion", self._smooth_selection))
        tb.addAction(self.a_undo); tb.addAction(self.a_redo); tb.addSeparator()

        # --- Herramientas (grupo exclusivo)
        self.tool_group = QtGui.QActionGroup(self)
        self.tool_group.setExclusive(True)
        m = mb.addMenu("&Herramientas")
        for mode, label, glyph, key, short in TOOLS:
            a = act(f"{glyph}  {label}", None, key, render.TOOL_HELP.get(mode), True)
            a.setIconText(f"{glyph} {short}")   # la barra usa la version corta
            a.setData(mode)
            a.triggered.connect(lambda _=False, md=mode: self._set_tool(md))
            self.tool_group.addAction(a)
            m.addAction(a)
            tb.addAction(a)
        tb.addSeparator()

        # --- Malla (reparaciones)
        m = mb.addMenu("&Malla")
        for label, fn in [
            ("Soldar nodos coincidentes…", self._weld),
            ("Eliminar elementos degenerados", lambda: self._repair("Eliminar degenerados", repair.remove_degenerate)),
            ("Eliminar elementos duplicados", lambda: self._repair("Eliminar duplicados", repair.remove_duplicate_elements)),
            ("Eliminar elementos de area nula", lambda: self._repair("Area nula", lambda me: repair.remove_tiny_area(me, 1.0))),
            ("Orientar todo a CCW", lambda: self._repair("Orientar CCW", repair.fix_orientation)),
            ("Conservar componente principal", lambda: self._repair("Componente principal", repair.keep_largest_component)),
            ("Hacer la frontera recorrible", lambda: self._repair("Frontera recorrible", repair.fix_nontraversable)),
            ("Eliminar elementos colgantes", lambda: self._repair("Colgantes", repair.remove_dangling)),
            ("Rellenar huecos internos falsos", lambda: self._repair("Rellenar huecos", repair.fill_holes)),
            ("Eliminar slivers de frontera", lambda: self._repair("Slivers de frontera", lambda me: repair.delete_boundary_slivers(me, 0.05))),
        ]:
            m.addAction(act(label, fn))
        m.addSeparator()
        m.addAction(act("Voltear aristas de baja calidad", lambda: self._repair("Voltear aristas", repair.flip_bad_edges)))
        m.addAction(act("Suavizar zonas de baja calidad…", self._smooth_bad))
        m.addSeparator()
        m.addAction(act("Renumerar (Cuthill-McKee inverso)", lambda: self._repair("Renumerar RCM", repair.renumber_rcm)))
        m.addAction(act("Compactar (purgar borrados)", lambda: self._repair("Compactar", repair.compact_mesh)))
        m.addSeparator()
        self.a_pipeline = act("Reparacion automatica completa…", self.run_pipeline)
        m.addAction(self.a_pipeline)

        # --- Calidad
        m = mb.addMenu("&Calidad")
        self.a_qc = act("Ejecutar control de calidad", lambda: self.run_qc(self.qc.params()), "F5")
        self.a_qc.setIconText("Calidad (F5)")
        m.addAction(self.a_qc)
        m.addAction(act("Exportar reporte…", self.export_report))
        tb.addAction(self.a_qc)

        # --- Fronteras
        m = mb.addMenu("&Fronteras")
        m.addAction(act("Generar automaticamente…", self.auto_boundaries))
        m.addAction(act("Crear nodestring desde la seleccion", lambda: self._boundary_from_selection(LAND, 20)))
        m.addAction(act("Borrar todas las fronteras", self._clear_boundaries))

        # --- Ver
        m = mb.addMenu("&Ver")
        m.addAction(act("Zoom a toda la malla", self.view.zoom_all, "Ctrl+0"))
        m.addAction(act("Zoom a la seleccion", self._zoom_selection, "Ctrl+E"))
        m.addSeparator()
        for d in (self.d_qc, self.d_stats, self.d_props, self.d_layers, self.d_bnd, self.d_log):
            m.addAction(d.toggleViewAction())

        m = mb.addMenu("A&yuda")
        m.addAction(act("Atajos y herramientas", self._help))
        m.addAction(act(f"Acerca de {APP_NAME}", self._about))

    def _build_status(self):
        sb = self.statusBar()
        self.lb_coord = QtWidgets.QLabel("—")
        self.lb_counts = QtWidgets.QLabel("—")
        self.lb_tool = QtWidgets.QLabel("")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(190); self.progress.setVisible(False)
        sb.addWidget(self.lb_tool, 1)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.lb_counts)
        sb.addPermanentWidget(self.lb_coord)
        self.view.sigStatus.connect(self.lb_tool.setText)

    # ------------------------------------------------------------- utilidades
    def log(self, msg):
        self.logpanel.log(msg)

    def _busy(self, on, text=""):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor) if on \
            else QtWidgets.QApplication.restoreOverrideCursor()
        if text:
            self.lb_tool.setText(text)
        QtWidgets.QApplication.processEvents()

    def _update_title(self):
        n = os.path.basename(self.path) if self.path else "sin titulo"
        dirty = "" if self.stack.isClean() else " •"
        self.setWindowTitle(f"{APP_NAME} — {n}{dirty} — {SUBTITLE}")
        if self.mesh is not None:
            self.lb_counts.setText(f"{self.mesh.n_nodes:,} nodos   "
                                   f"{self.mesh.n_elems:,} elementos")

    def _after_change(self):
        self.view.refresh()
        self.bnd.set_mesh(self.mesh)
        self._update_title()
        self.props.update_view(self.mesh, self.view.sel_nodes, self.view.sel_elems)

    def _push(self, op, text=None):
        if push(self.stack, op, self._after_change, text):
            self._after_change()
            return True
        return False

    def _need_mesh(self):
        if self.mesh is None:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Primero abre una malla.")
            return False
        return True

    # ------------------------------------------------------------------ E/S
    def open_file(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir malla", "",
            "Mallas (*.14 *.grd *.2dm);;fort.14 ADCIRC (*.14 *.grd);;SMS 2dm (*.2dm);;Todos (*)")
        if not p:
            return
        self.load(p)

    def load(self, p):
        self._busy(True, f"Leyendo {os.path.basename(p)}…")
        t0 = time.time()
        try:
            m = load_mesh(p, progress=lambda s: self.log("  " + s))
        except Exception as exc:
            self._busy(False)
            self.log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"No se pudo leer:\n{exc}")
            return
        self.mesh = m
        self.path = p
        self.stack.clear()
        self.report = None
        self.qc.set_report(None); self.stats.set_report(None); self.layers.set_report(None)
        self.view.set_mesh(m)
        self.bnd.set_mesh(m)
        self.props.update_view(m, [], [])
        self._busy(False)
        self._update_title()
        self.log(f"Abierto {p}")
        self.log(f"  {m.n_nodes:,} nodos, {m.n_elems:,} elementos, "
                 f"{len(m.boundaries)} fronteras  ({time.time() - t0:.2f} s)")
        x0, y0, x1, y1 = m.bbox()
        self.log(f"  extension: lon [{x0:.4f}, {x1:.4f}]  lat [{y0:.4f}, {y1:.4f}]")
        self.run_qc(self.qc.params())

    def save_file(self):
        if not self._need_mesh():
            return
        if not self.path:
            return self.save_file_as()
        self._write(self.path)

    def save_file_as(self):
        if not self._need_mesh():
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Guardar malla como", self.path or "malla.14",
            "fort.14 ADCIRC (*.14);;SMS 2dm (*.2dm)")
        if p:
            self.path = p
            self._write(p)

    def _write(self, p):
        self._busy(True, f"Guardando {os.path.basename(p)}…")
        try:
            save_mesh(p, self.mesh, progress=lambda s: self.log("  " + s))
        except Exception as exc:
            self._busy(False)
            self.log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"No se pudo guardar:\n{exc}")
            return
        self._busy(False)
        self.stack.setClean()
        self._update_title()
        self.log(f"Guardado en {p}")

    def export_report(self):
        if self.report is None:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Ejecuta primero el control de calidad.")
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exportar reporte", "reporte_qc.txt",
                                                     "Texto (*.txt)")
        if not p:
            return
        with open(p, "w") as f:
            f.write(f"{APP_NAME} — reporte de calidad\n")
            f.write(f"malla: {self.path}\n")
            f.write(f"nodos: {self.mesh.n_nodes}   elementos: {self.mesh.n_elems}\n\n")
            f.write("PROBLEMAS\n")
            for c in self.report.checks:
                mark = {"error": "XX", "warning": "!!", "info": "..", "ok": "OK"}[c.severity]
                f.write(f"  {mark} {c.label:.<62s} {c.count}\n")
                if c.detail:
                    f.write(f"        {c.detail}\n")
            f.write("\nESTADISTICAS\n")
            for k, s in self.report.stats.items():
                f.write(f"  {k:<26s} min={s['min']:12.4g} p1={s['p1']:12.4g} "
                        f"med={s['med']:12.4g} p99={s['p99']:12.4g} max={s['max']:12.4g}"
                        f" {s['unit']}\n")
        self.log(f"Reporte exportado a {p}")

    # ----------------------------------------------------------------- QC
    def run_qc(self, params=None):
        if self.mesh is None:
            return
        params = params or self.qc.params()
        self._busy(True, "Ejecutando control de calidad…")
        t0 = time.time()
        try:
            rep = run_qc(self.mesh, progress=lambda s: None, **params)
        except Exception as exc:
            self._busy(False); self.log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Fallo el QC:\n{exc}")
            return
        self.report = rep
        self.qc.set_report(rep)
        self.stats.set_report(rep)
        self.layers.set_report(rep)
        self._busy(False)
        from .panels import plural
        resumen = f"{plural(rep.n_errors, 'error', 'errores')}, " \
                  f"{plural(rep.n_warnings, 'aviso', 'avisos')}"
        self.log(f"QC: {resumen}  ({time.time() - t0:.2f} s)")
        self.lb_tool.setText(f"QC completado: {resumen}")

    def _hl_from_qc(self, kind, ids):
        if kind == NODE:
            self.view.set_highlight(nodes=ids)
        elif kind == ELEM:
            self.view.set_highlight(elems=ids)
        else:
            self.view.set_highlight()

    def _select_from_qc(self, kind, ids):
        if kind == NODE:
            self.view.set_selection(nodes=ids)
        else:
            self.view.set_selection(elems=ids)

    def _zoom_check(self, kind, ids):
        """Zoom sobre los casos de un chequeo (nodos o elementos)."""
        ids = np.atleast_1d(np.asarray(ids, np.int64))
        if self.mesh is None or not len(ids):
            return
        nodes = np.unique(self.mesh.tri[ids].ravel()) if kind == ELEM else ids
        self.view.zoom_to_nodes(nodes, pad=0.9 if len(ids) == 1 else 0.25)
        if kind == ELEM:
            self.view.set_highlight(elems=ids)
        else:
            self.view.set_highlight(nodes=ids)

    def _zoom_nodes(self, ids):
        if self.mesh is not None:
            self.view.zoom_to_nodes(np.atleast_1d(np.asarray(ids, np.int64)))

    # ------------------------------------------------------------ herramientas
    def _set_tool(self, mode):
        self.view.set_tool(mode)
        for a in self.tool_group.actions():
            a.setChecked(a.data() == mode)
        self._pending = []
        self.view.set_pending([])

    def _on_cursor(self, x, y):
        txt = f"lon {x:.6f}°   lat {y:.6f}°"
        if self.mesh is not None:
            i = self.mesh.pick_node(x, y)
            if i is not None:
                txt += f"   nodo {i + 1}  z = {self.mesh.z[i]:.2f} m"
        self.lb_coord.setText(txt)

    def _on_selection(self):
        self.props.update_view(self.mesh, self.view.sel_nodes, self.view.sel_elems)

    def _on_tool_action(self, mode, payload, mods):
        if self.mesh is None:
            return
        try:
            self._dispatch(mode, payload, mods)
        except Exception:
            self.log(traceback.format_exc())

    def _dispatch(self, mode, payload, mods):
        m = self.mesh
        kind = payload[0]

        if kind == "box":
            _, (x0, y0), (x1, y1) = payload
            self._box_select(mode, x0, y0, x1, y1, mods)
            return

        if kind == "drag" and mode == render.MOVE_NODE:
            _, start, cur, state = payload
            if state == "start":
                i = m.pick_node(*start)
                self._drag_node = i
                # posicion original del NODO (no la del cursor): es el delta a deshacer
                self._drag_from = None if i is None else m.xy[i].copy()
            if self._drag_node is None:
                return
            i = self._drag_node
            if state in ("start", "move"):
                m.xy[i] = cur                    # vista previa en vivo
                m.touch_geometry()
                self.view.refresh()
            else:
                m.xy[i] = self._drag_from        # revertir antes de grabar el delta
                m.touch_geometry()
                self._push(edit.move_nodes(m, [i], [np.asarray(cur, float)]),
                           f"Mover nodo {i + 1}")
                self._drag_node = self._drag_from = None
            return

        _, (x, y) = payload

        if mode in (render.SEL_NODE, render.SEL_ELEM):
            self._click_select(mode, x, y, mods)
        elif mode == render.ADD_NODE:
            self._push(edit.add_node(m, x, y), "Anadir nodo")
        elif mode == render.DEL_NODE:
            i = m.pick_node(x, y)
            if i is not None:
                self._push(edit.delete_nodes(m, [i]), "Borrar nodo")
        elif mode == render.DEL_ELEM:
            k = m.pick_element(x, y)
            if k is not None:
                self._push(edit.delete_elements(m, [k]), "Borrar elemento")
        elif mode == render.SPLIT_EDGE:
            e = m.pick_edge(x, y)
            if e:
                self._push(edit.split_edge(m, *e), "Dividir arista")
        elif mode == render.SWAP_EDGE:
            e = m.pick_edge(x, y)
            if e:
                op = edit.swap_edge(m, *e)
                if op is None:
                    self.lb_tool.setText("No se puede voltear: el cuadrilatero no es convexo "
                                         "o la arista esta en la frontera")
                else:
                    self._push(op, "Voltear arista")
        elif mode in (render.CREATE_ELEM, render.MERGE_NODES):
            i = m.pick_node(x, y)
            if i is None:
                return
            self._pending.append(i)
            self.view.set_pending(self._pending)
            need = 3 if mode == render.CREATE_ELEM else 2
            if len(self._pending) >= need:
                if mode == render.CREATE_ELEM:
                    op = edit.create_element(m, *self._pending[:3])
                    txt = "Crear elemento"
                else:
                    op = edit.merge_nodes(m, self._pending[0], self._pending[1])
                    txt = "Fusionar nodos"
                self._pending = []
                self.view.set_pending([])
                if op is None:
                    self.lb_tool.setText("Operacion no valida sobre esos nodos")
                else:
                    self._push(op, txt)

    def _click_select(self, mode, x, y, mods):
        m = self.mesh
        add = bool(mods & QtCore.Qt.ShiftModifier)
        sub = bool(mods & QtCore.Qt.ControlModifier)
        if mode == render.SEL_NODE:
            i = m.pick_node(x, y)
            cur = self.view.sel_nodes
            if i is None:
                return
            new = self._combine(cur, [i], add, sub)
            self.view.set_selection(nodes=new)
        else:
            k = m.pick_element(x, y)
            if k is None:
                return
            new = self._combine(self.view.sel_elems, [k], add, sub)
            self.view.set_selection(elems=new)

    def _box_select(self, mode, x0, y0, x1, y1, mods):
        m = self.mesh
        add = bool(mods & QtCore.Qt.ShiftModifier)
        sub = bool(mods & QtCore.Qt.ControlModifier)
        xa, xb = sorted((x0, x1)); ya, yb = sorted((y0, y1))
        if mode == render.SEL_NODE:
            live = m.live_nodes()
            p = m.xy[live]
            hit = live[(p[:, 0] >= xa) & (p[:, 0] <= xb) &
                       (p[:, 1] >= ya) & (p[:, 1] <= yb)]
            self.view.set_selection(nodes=self._combine(self.view.sel_nodes, hit, add, sub))
        else:
            ids, t = m.live_elems()
            c = m.xy[t].mean(axis=1)
            hit = ids[(c[:, 0] >= xa) & (c[:, 0] <= xb) &
                      (c[:, 1] >= ya) & (c[:, 1] <= yb)]
            self.view.set_selection(elems=self._combine(self.view.sel_elems, hit, add, sub))

    @staticmethod
    def _combine(cur, new, add, sub):
        new = np.atleast_1d(np.asarray(new, np.int64))
        if add:
            return np.union1d(cur, new)
        if sub:
            return np.setdiff1d(cur, new)
        return new

    # -------------------------------------------------------------- seleccion
    def _select_all_nodes(self):
        if self._need_mesh():
            self.view.set_selection(nodes=self.mesh.live_nodes())

    def _clear_sel(self):
        self.view.clear_selection()
        self.view.set_highlight()

    def _invert_sel(self):
        if not self._need_mesh():
            return
        if len(self.view.sel_nodes):
            self.view.set_selection(nodes=np.setdiff1d(self.mesh.live_nodes(),
                                                       self.view.sel_nodes))
        elif len(self.view.sel_elems):
            ids, _ = self.mesh.live_elems()
            self.view.set_selection(elems=np.setdiff1d(ids, self.view.sel_elems))

    def _zoom_selection(self):
        if not self._need_mesh():
            return
        if len(self.view.sel_nodes):
            self.view.zoom_to_nodes(self.view.sel_nodes)
        elif len(self.view.sel_elems):
            self.view.zoom_to_nodes(np.unique(self.mesh.tri[self.view.sel_elems].ravel()))

    def _delete_selection(self):
        if not self._need_mesh():
            return
        if len(self.view.sel_elems):
            self._push(edit.delete_elements(self.mesh, self.view.sel_elems),
                       f"Borrar {len(self.view.sel_elems)} elementos")
        elif len(self.view.sel_nodes):
            self._push(edit.delete_nodes(self.mesh, self.view.sel_nodes),
                       f"Borrar {len(self.view.sel_nodes)} nodos")
        self.view.clear_selection()

    def _smooth_selection(self):
        if not self._need_mesh():
            return
        ids = self.view.sel_nodes
        if not len(ids) and len(self.view.sel_elems):
            ids = np.unique(self.mesh.tri[self.view.sel_elems].ravel())
        if not len(ids):
            QtWidgets.QMessageBox.information(self, APP_NAME, "Selecciona nodos o elementos.")
            return
        n, ok = QtWidgets.QInputDialog.getInt(self, "Suavizar", "Iteraciones:", 20, 1, 500)
        if not ok:
            return
        self._busy(True, "Suavizando…")
        op = edit.smooth_nodes(self.mesh, ids, iters=n)
        self._busy(False)
        if not self._push(op, "Suavizar"):
            self.lb_tool.setText("El suavizado no encontro mejoras")

    def _apply_depth(self, ids, value):
        if self._need_mesh() and len(ids):
            self._push(edit.set_depths(self.mesh, ids, value),
                       f"Profundidad = {value:g} m en {len(ids)} nodos")

    def _apply_xy(self, i, x, y):
        if self._need_mesh():
            self._push(edit.move_nodes(self.mesh, [i], [[x, y]]), "Mover nodo")

    # ---------------------------------------------------------------- capas
    def _apply_layers(self, d):
        v = self.view
        v.wire_lod = d.get("lod", True)
        v.show_wire = d["wire"]; v.show_nodes = d["nodes"]
        v.show_bnd = d["bnd"]; v.show_ns = d["ns"]
        v.set_show_fill(d["fill"])
        v.refresh(full=True)

    def _apply_field(self, spec):
        if spec is None or self.report is None:
            self.view.set_field(None, None, "node")
            return
        name, cmap, vmin, vmax = spec
        vals, kind = self.report.fields[name]
        self.view.set_field(name, vals, kind, cmap, (vmin, vmax))

    # ------------------------------------------------------------ reparacion
    def _repair(self, name, fn, refresh_qc=True):
        if not self._need_mesh():
            return
        self._busy(True, f"{name}…")
        before = self.mesh.snapshot()
        t0 = time.time()
        try:
            msg = fn(self.mesh)
        except Exception as exc:
            self._busy(False); self.log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"{name} fallo:\n{exc}")
            return
        after = self.mesh.snapshot()
        op = edit.SnapshotOp(self.mesh, name, before, after)
        self.stack.push(_snap_cmd(op, self._after_change, name))
        self._busy(False)
        self.log(f"[{name}] {msg}  ({time.time() - t0:.2f} s)")
        self.lb_tool.setText(f"{name}: {msg}")
        self.view.clear_selection()
        self._after_change()
        if refresh_qc and self.report is not None:
            self.run_qc(self.qc.params())

    def _weld(self):
        tol, ok = QtWidgets.QInputDialog.getDouble(
            self, "Soldar nodos", "Distancia maxima (m):", 5.0, 0.001, 10000, 3)
        if ok:
            self._repair("Soldar nodos", lambda m: repair.weld_nodes(m, tol))

    def _smooth_bad(self):
        q, ok = QtWidgets.QInputDialog.getDouble(
            self, "Suavizar zonas malas", "Suavizar alrededor de elementos con calidad <:",
            0.30, 0.01, 0.95, 2)
        if ok:
            self._repair("Suavizado local", lambda m: repair.smooth_bad_regions(m, q))

    def run_pipeline(self):
        if not self._need_mesh():
            return
        dlg = _PipelineDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        steps = dlg.selected()
        if not steps:
            return
        before = self.mesh.snapshot()
        self.progress.setVisible(True); self.progress.setRange(0, len(repair.PIPELINE))
        self._busy(True, "Reparacion automatica…")
        t0 = time.time()

        def prog(i, n, name):
            self.progress.setValue(i)
            self.lb_tool.setText(f"[{i + 1}/{n}] {name}")
            QtWidgets.QApplication.processEvents()

        try:
            out = repair.run_pipeline(self.mesh, steps, prog)
        except Exception as exc:
            self._busy(False); self.progress.setVisible(False)
            self.log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"La reparacion fallo:\n{exc}")
            return
        after = self.mesh.snapshot()
        self.stack.push(_snap_cmd(edit.SnapshotOp(self.mesh, "Reparacion automatica",
                                                  before, after),
                                  self._after_change, "Reparacion automatica"))
        self.progress.setVisible(False)
        self._busy(False)
        self.log("── Reparacion automatica ──")
        self.log(out)
        self.log(f"── total {time.time() - t0:.1f} s ──")
        self.view.clear_selection()
        self.view.set_mesh(self.mesh, reset_view=False)
        self._after_change()
        self.run_qc(self.qc.params())

    # ------------------------------------------------------------- fronteras
    def auto_boundaries(self):
        if not self._need_mesh():
            return
        dlg = _AutoBndDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        p = dlg.values()
        before = self.mesh.snapshot()
        self._busy(True, "Generando fronteras…")
        msg = repair.auto_boundaries(self.mesh, **p)
        after = self.mesh.snapshot()
        self.stack.push(_snap_cmd(edit.SnapshotOp(self.mesh, "Fronteras automaticas",
                                                  before, after),
                                  self._after_change, "Fronteras automaticas"))
        self._busy(False)
        self.log(f"[Fronteras] {msg}")
        self.lb_tool.setText(msg)
        self._after_change()

    def _boundary_from_selection(self, kind, ibtype):
        if not self._need_mesh():
            return
        ids = self.view.sel_nodes
        if len(ids) < 2:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "Selecciona al menos 2 nodos con la herramienta de nodos.")
            return
        ordered = _order_along_boundary(self.mesh, ids)
        r = edit.Recorder(self.mesh, "Crear nodestring")
        b = Boundary(kind, ordered, None if kind == OCEAN else ibtype,
                     f"{'Oceano' if kind == OCEAN else 'Tierra' if kind == LAND else 'Isla'} "
                     f"{len(self.mesh.boundaries) + 1}")
        r.set_boundaries(list(self.mesh.boundaries) + [b])
        self._push(r.finish(), "Crear nodestring")
        self.log(f"[Frontera] nodestring de {len(ordered)} nodos ({kind})")

    def _delete_boundaries(self, idx):
        if not self._need_mesh():
            return
        keep = [b for i, b in enumerate(self.mesh.boundaries) if i not in set(idx)]
        r = edit.Recorder(self.mesh, "Borrar fronteras")
        r.set_boundaries(keep)
        self._push(r.finish(), "Borrar fronteras")

    def _set_boundary_type(self, i, kind, ibtype):
        if not self._need_mesh():
            return
        bs = [b.copy() for b in self.mesh.boundaries]
        bs[i].kind = kind
        bs[i].ibtype = None if kind == OCEAN else ibtype
        r = edit.Recorder(self.mesh, "Cambiar tipo de frontera")
        r.set_boundaries(bs)
        self._push(r.finish(), "Cambiar tipo de frontera")

    def _clear_boundaries(self):
        if not self._need_mesh():
            return
        if QtWidgets.QMessageBox.question(self, APP_NAME,
                                          "¿Borrar todas las fronteras?") != \
                QtWidgets.QMessageBox.Yes:
            return
        r = edit.Recorder(self.mesh, "Borrar fronteras")
        r.set_boundaries([])
        self._push(r.finish(), "Borrar fronteras")

    # ----------------------------------------------------------------- ayuda
    def _help(self):
        rows = "".join(f"<tr><td style='padding:2px 14px 2px 0'><b>{k}</b></td>"
                       f"<td>{v}</td></tr>"
                       for k, v in [
                           ("Ctrl+O / Ctrl+S", "abrir / guardar"),
                           ("Ctrl+Z / Ctrl+Shift+Z", "deshacer / rehacer"),
                           ("F5", "ejecutar control de calidad"),
                           ("Ctrl+0", "zoom a toda la malla"),
                           ("Ctrl+E", "zoom a la seleccion"),
                           ("Ctrl+A / Ctrl+D", "seleccionar todo / limpiar"),
                           ("Supr", "borrar la seleccion"),
                           ("Rueda", "zoom"),
                           ("Boton derecho", "zoom con arrastre"),
                           ("Shift + clic", "sumar a la seleccion"),
                           ("Ctrl + clic", "restar de la seleccion"),
                       ])
        tools = "".join(f"<tr><td style='padding:2px 14px 2px 0'><b>{k}</b> "
                        f"{lab}</td><td>{render.TOOL_HELP[md]}</td></tr>"
                        for md, lab, gl, k, _ in TOOLS)
        QtWidgets.QMessageBox.information(
            self, "Atajos y herramientas",
            f"<h3>Atajos</h3><table>{rows}</table>"
            f"<h3>Herramientas</h3><table>{tools}</table>")

    def _about(self):
        QtWidgets.QMessageBox.about(
            self, f"Acerca de {APP_NAME}",
            f"<h2>{APP_NAME}</h2><p>{SUBTITLE}</p>"
            "<p>Edicion manual de mallas triangulares no estructuradas con "
            "control de calidad orientado a ADCIRC y SWAN: topologia, forma, "
            "criterio CFL, batimetria y fronteras (nodestrings con IBTYPE).</p>"
            "<p>Formatos: fort.14 (ADCIRC) y .2dm (SMS Aquaveo).</p>")

    def closeEvent(self, ev):
        if self.mesh is not None and not self.stack.isClean():
            r = QtWidgets.QMessageBox.question(
                self, APP_NAME, "Hay cambios sin guardar. ¿Guardar antes de salir?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard |
                QtWidgets.QMessageBox.Cancel)
            if r == QtWidgets.QMessageBox.Cancel:
                ev.ignore(); return
            if r == QtWidgets.QMessageBox.Save:
                self.save_file()
        ev.accept()


def _snap_cmd(op, on_change, text):
    from .commands import OpCommand
    return OpCommand(op, on_change, text)


def _order_along_boundary(mesh, ids):
    """Ordena un conjunto de nodos siguiendo la cadena de aristas de frontera."""
    ids = np.unique(np.asarray(ids, np.int64))
    sel = set(int(v) for v in ids)
    be = mesh.boundary_edges()
    adj = {}
    for a, b in be:
        a, b = int(a), int(b)
        if a in sel and b in sel:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
    if not adj:
        return ids
    ends = [k for k, v in adj.items() if len(v) == 1]
    start = ends[0] if ends else next(iter(adj))
    out, seen, cur, prev = [start], {start}, start, None
    while True:
        nxt = [k for k in adj.get(cur, []) if k != prev and k not in seen]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        out.append(cur); seen.add(cur)
    rest = [int(v) for v in ids if int(v) not in seen]
    return np.asarray(out + rest, np.int64)


class _PipelineDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Reparacion automatica")
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(
            "Se ejecutan en orden. Todo el bloque se deshace con un solo Ctrl+Z."))
        self.boxes = []
        for name, _ in repair.PIPELINE:
            c = QtWidgets.QCheckBox(name); c.setChecked(True)
            v.addWidget(c); self.boxes.append(c)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def selected(self):
        return [c.text() for c in self.boxes if c.isChecked()]


class _AutoBndDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Generar fronteras automaticamente")
        f = QtWidgets.QFormLayout(self)
        self.edge = QtWidgets.QDoubleSpinBox(); self.edge.setRange(1, 1e6)
        self.edge.setValue(3000); self.edge.setSuffix(" m")
        self.dep = QtWidgets.QDoubleSpinBox(); self.dep.setRange(0, 1e5)
        self.dep.setValue(200); self.dep.setSuffix(" m")
        self.land = QtWidgets.QSpinBox(); self.land.setRange(0, 99); self.land.setValue(20)
        self.isl = QtWidgets.QSpinBox(); self.isl.setRange(0, 99); self.isl.setValue(21)
        f.addRow("Arista minima para considerar oceano", self.edge)
        f.addRow("Profundidad minima para considerar oceano", self.dep)
        f.addRow("IBTYPE de tierra firme", self.land)
        f.addRow("IBTYPE de islas", self.isl)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        f.addRow(bb)

    def values(self):
        return dict(min_edge_m=self.edge.value(), min_depth=self.dep.value(),
                    land_ibtype=self.land.value(), island_ibtype=self.isl.value())

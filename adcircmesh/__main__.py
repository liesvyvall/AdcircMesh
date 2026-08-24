"""Punto de entrada: python -m adcircmesh [malla.14]"""
import sys
import warnings

warnings.filterwarnings("ignore")


def apply_theme(app):
    """Estilo Fusion con paleta clara; el resto lo pone la hoja de estilo."""
    from PySide6 import QtGui

    app.setStyle("Fusion")
    f = app.font(); f.setPointSize(max(f.pointSize(), 13)); app.setFont(f)
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#f4f5f7"))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#1f2430"))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#f7f8fa"))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#1f2430"))
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#1f2430"))
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#2563eb"))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("#1f2430"))
    app.setPalette(pal)


def main():
    from PySide6 import QtWidgets
    from .gui.main_window import MainWindow, APP_NAME

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    apply_theme(app)

    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.load(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

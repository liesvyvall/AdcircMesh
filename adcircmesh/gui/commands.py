"""Envoltorios QUndoCommand sobre las operaciones del nucleo."""
from __future__ import annotations

from PySide6 import QtGui


class OpCommand(QtGui.QUndoCommand):
    """La operacion ya viene aplicada; el primer redo() de Qt no debe repetirla."""

    def __init__(self, op, on_change=None, text=None):
        super().__init__(text or getattr(op, "name", "Editar"))
        self.op = op
        self._first = True
        self._on_change = on_change

    def redo(self):
        if self._first:
            self._first = False
        else:
            self.op.redo()
        if self._on_change:
            self._on_change()

    def undo(self):
        self.op.undo()
        if self._on_change:
            self._on_change()


def push(stack, op, on_change=None, text=None):
    """Empuja una operacion si no esta vacia. Devuelve True si hubo cambio."""
    if not op:
        return False
    stack.push(OpCommand(op, on_change, text))
    return True

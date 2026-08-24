#!/usr/bin/env python
"""Lanzador de AdcircMesh:  python adcircmesh.py [malla.14]"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from adcircmesh.__main__ import main

main()

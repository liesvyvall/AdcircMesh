#!/usr/bin/env python
"""Descarga una malla ADCIRC de ejemplo para probar AdcircMesh.

La malla NO se incluye en este repositorio: `adcirc/adcirc-testsuite` se
distribuye sin fichero de licencia, asi que se descarga bajo demanda en lugar
de redistribuirla.  El caso `shinnecock_inlet` es el ejemplo canonico de
ADCIRC y de ADCIRC+SWAN acoplado (Shinnecock Inlet, Long Island, Nueva York):
5.780 nodos y 10.798 elementos, unos 340 kB.

Uso:
    python examples/get_example.py                # descarga shinnecock_inlet
    python examples/get_example.py quarterannular # otro caso disponible
"""
import pathlib
import sys
import urllib.request

BASE = ("https://raw.githubusercontent.com/adcirc/adcirc-testsuite/"
        "main/adcirc/{case}/fort.14")

CASES = {
    "shinnecock_inlet": "adcirc_shinnecock_inlet",
    "quarterannular": "adcirc_quarterannular-2d",
    "alaska_ice": "adcirc_alaska_ice-2d",
    "global_tide": "adcirc_global-tide-2d",
}


def fetch(name="shinnecock_inlet"):
    if name not in CASES:
        sys.exit(f"caso desconocido: {name}\ndisponibles: {', '.join(CASES)}")
    out = pathlib.Path(__file__).parent / f"{name}.14"
    url = BASE.format(case=CASES[name])
    print(f"descargando {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except Exception as exc:
        sys.exit(f"fallo la descarga: {exc}")
    out.write_bytes(data)
    print(f"guardado en {out}  ({len(data) / 1024:.0f} kB)")
    print(f"\nabrelo con:\n    python adcircmesh.py {out}")
    return out


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else "shinnecock_inlet")

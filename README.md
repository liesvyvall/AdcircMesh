# AdcircMesh

**Editor visual de mallas no estructuradas con control de calidad para ADCIRC y SWAN.**

Aplicación de escritorio en Python para editar a mano mallas triangulares no
estructuradas y aplicarles los chequeos que exigen ADCIRC y SWAN antes de
correr un modelo. Es una alternativa libre al flujo de trabajo de SMS (Aquaveo)
para esta tarea concreta.

![Vista general](docs/captura-general.png)

## Instalación

```bash
git clone https://github.com/liesvyvall/AdcircMesh.git
cd AdcircMesh
pip install -r requirements.txt
```

O como paquete instalable:

```bash
pip install -e .
adcircmesh malla.14
```

Requiere Python ≥ 3.10. Probado en macOS (Apple Silicon); PySide6 y pyqtgraph
funcionan igual en Linux y Windows.

## Uso

```bash
python examples/get_example.py        # descarga una malla de prueba
python adcircmesh.py examples/shinnecock_inlet.14
```

La malla de ejemplo **no viene en el repositorio**: se descarga bajo demanda del
[ADCIRC test suite](https://github.com/adcirc/adcirc-testsuite), que se
distribuye sin fichero de licencia. El caso `shinnecock_inlet` (Shinnecock
Inlet, Long Island, Nueva York) es el ejemplo canónico de ADCIRC y de
ADCIRC+SWAN acoplado: 3.070 nodos y 5.780 elementos.

## Qué hace

### Formatos

Lee y escribe `fort.14` de ADCIRC con sus secciones completas
(NOPE / NETA / NVDLL / NBOU / NVEL / IBTYPE, incluidos los datos extra por nodo
de las barreras) y `.2dm` de SMS, convirtiendo elevación ↔ profundidad. El ida
y vuelta es exacto en coordenadas, conectividad y fronteras.

### Edición manual

| Tecla | Herramienta | Qué hace |
|---|---|---|
| `Esc` | Navegar | desplazar y zoom |
| `N` / `E` | Seleccionar nodos / elementos | clic o rectángulo; `Shift` suma, `Ctrl` resta |
| `M` | Mover nodo | arrastrar con vista previa en vivo |
| `A` | Añadir nodo | subdivide en 3 el elemento donde caes, interpolando la batimetría por coordenadas baricéntricas |
| `D` / `X` | Borrar nodo / elemento | |
| `S` | Dividir arista | inserta el punto medio y parte los elementos incidentes |
| `F` | Voltear arista | comprueba convexidad antes de aplicar |
| `C` | Crear elemento | clic en 3 nodos |
| `G` | Fusionar nodos | colapsa el segundo sobre el primero |

Todo pasa por una pila de deshacer: **`Ctrl+Z` / `Ctrl+Shift+Z` revierten
cualquier cosa**, incluida una reparación automática completa como un solo paso.

### Control de calidad (`F5`)

![Control de calidad](docs/captura-calidad.png)

34 chequeos agrupados en cinco familias. Cada uno guarda los **índices
infractores**: al seleccionarlo se resaltan en el mapa, con doble clic haces
zoom, y los botones `◀ ▶` recorren los casos uno por uno.

- **Integridad** — índices fuera de rango, nodos repetidos, nodos y elementos duplicados, huérfanos
- **Topología** — aristas no-manifold, frontera no recorrible, componentes desconectadas, elementos colgantes, valencia
- **Geometría** — orientación CW (ADCIRC exige CCW), área nula, slivers, ángulo mínimo y máximo, elementos con sus tres nodos en la frontera, gradación de tamaño
- **ADCIRC / SWAN** — violación de CFL para un `dt` objetivo, profundidades ≤ 0 y < −10 m, NaN, saltos batimétricos bruscos entre nodos vecinos
- **Fronteras** — NOPE = 0, nodos fuera de rango, declarados que no están en el borde real, nodos del borde real sin declarar, solapes entre nodestrings

Los umbrales (paso de tiempo, ángulo mínimo, valencia, profundidad, gradación)
se ajustan en el panel. El reporte se exporta a texto.

### Color por campo

Relleno continuo por profundidad, calidad de forma, ángulo mínimo o máximo,
área, arista mínima, `dt` máximo por CFL, gradación o valencia, con paleta y
rango configurables.

### Reparación

Menú **Malla**: soldar nodos coincidentes, eliminar degenerados, duplicados y
área nula, orientar a CCW, conservar la componente principal, hacer la frontera
recorrible, quitar elementos colgantes, rellenar huecos internos falsos, voltear
aristas de baja calidad, suavizado laplaciano local con garantía de no invertir
elementos y con la costa deslizando tangencialmente, eliminar slivers de
frontera, **renumerar con Cuthill-McKee inverso** y compactar. Hay además un
pipeline automático con las etapas seleccionables.

El renumerado RCM importa más de lo que parece: en una malla del Pacífico
mexicano de 243.315 elementos bajó el ancho de banda de 124.609 a 901.

### Fronteras (nodestrings)

![Edición](docs/captura-edicion.png)

Panel con la lista de fronteras, su tipo y su IBTYPE. Se pueden generar
automáticamente (detecta el tramo oceánico por profundidad y tamaño de arista,
lo extiende hasta la costa y marca las islas), crear desde una selección de
nodos (que se ordenan siguiendo la cadena de aristas de frontera), cambiarles el
tipo o borrarlas.

## Arquitectura

```
adcircmesh/
  core/          # sin dependencia de Qt: usable desde script
    mesh.py      # clase Mesh, proyección CPP, adyacencias CSR, picking
    meshio.py    # fort.14 y 2dm
    quality.py   # motor de QC, devuelve los índices infractores
    edit.py      # operaciones de edición reversibles
    repair.py    # reparaciones globales y fronteras automáticas
  gui/
    render.py    # lienzo pyqtgraph + ráster Agg para el relleno
    panels.py    # paneles acoplables
    commands.py  # puente con la pila de deshacer de Qt
    main_window.py
```

El núcleo no importa Qt, así que sirve igual sin interfaz:

```python
from adcircmesh.core.meshio import load_mesh, save_mesh
from adcircmesh.core.quality import run_qc
from adcircmesh.core import repair

m = load_mesh("malla.14")
print(repair.renumber_rcm(m))

rep = run_qc(m, dt_target=1.0, min_angle=30.0)
print(rep.n_errors, "errores")
print(rep.by_key("angle_10").ids)      # elementos con ángulo mínimo < 10°

save_mesh("salida.14", m)
```

### Dos decisiones que explican el rendimiento

**Nada se borra de verdad durante la edición.** Los nodos y elementos se marcan
muertos y solo se compactan al guardar. Los índices nunca se invalidan, así que
deshacer es un delta pequeño en lugar de una copia de la malla entera.

**Dos motores de dibujo, porque difieren 400×.** El wireframe completo de una
malla de 368.000 aristas se dibuja en ~25 ms con un único `QPainterPath`; ese
mismo trazado *relleno* tarda 10,8 s, porque rellenar cientos de miles de
subtrazos es patológico para `QPainter`. Por eso el color va por un ráster Agg
fuera de pantalla (~0,6 s, con retardo) que se muestra como imagen. Al alejarse,
las aristas caen por debajo de 2 px y se ocultan solas; los marcadores de nodo
aparecen solo cuando su separación en pantalla supera los 14 px.

## Licencia

MIT — ver [LICENSE](LICENSE).

Una nota sobre las dependencias: **PySide6 es LGPLv3**, no MIT. Al usarse como
biblioteca dinámica desde Python eso es compatible con distribuir este código
bajo MIT, siempre que quien reciba un binario empaquetado pueda sustituir la
biblioteca Qt. Fue una de las razones para elegir PySide6 sobre PyQt5, que es
GPL. `pyqtgraph` es MIT; `numpy`, `scipy` y `matplotlib` son BSD.

## Créditos

- [ADCIRC](https://adcirc.org) — Luettich (UNC Chapel Hill) y Westerink (Notre Dame)
- [SWAN](https://swanmodel.sourceforge.io) — TU Delft
- Malla de ejemplo: [adcirc/adcirc-testsuite](https://github.com/adcirc/adcirc-testsuite)

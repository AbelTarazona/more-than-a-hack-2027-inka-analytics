"""Tercera version: la v2 con la velocidad del balon calculada para decidir quien controla.

La columna de velocidad del tracking falta en mas del 90 % de los frames con balon visible y nunca
supera 7,3 m/s, asi que la regla "un balon a mas de 10 m/s viaja, no esta controlado" no actuaba
nunca: cualquier pase que pasaba cerca de un jugador contaba como control suyo. Aqui la velocidad
se calcula con las posiciones del balon.

Esa velocidad solo se usa en el control, donde pesa frame a frame. El juego parado sigue igual que
en la v2 porque sus reglas se disparan con un solo frame rapido, y el ruido de posicion las
activaria en falso.

Descartado: rellenar huecos con el balon de broadcast. La sincronia funciona, pero solo gana unos 4
puntos de cobertura y no mejora los examenes. Se conserva como examen independiente en
solution/compare_versions.py.

    python solution/possession_v3.py
    python evaluation/validate_submission.py evaluation/results/inka_analytics_v3.csv
"""

import argparse
import dataclasses
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from broadcast_ball import with_derived_speed  # noqa: E402
from possession_v1 import FPS, GAME, ROOT, write_submission  # noqa: E402
from possession_v2 import decode, frame_score, load_match  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "inka_analytics_v3.csv")


def with_control_speed(match):
    """Partido cuyo balon lleva la velocidad calculada. El juego parado ya viene decidido."""
    return dataclasses.replace(match, ball=with_derived_speed(match.ball, FPS))


def predict(game=GAME):
    match = with_control_speed(load_match(game))
    score, summary = frame_score(match, game)
    print("  " + "  ".join("{} {}".format(key, value) for key, value in summary.items()))
    return decode(match, score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="CSV de salida")
    args = parser.parse_args()

    result = predict()
    path = write_submission(args.out, result)
    counts = {label: int(np.sum(result == label)) for label in "HADX"}
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("escrito en {}".format(path))

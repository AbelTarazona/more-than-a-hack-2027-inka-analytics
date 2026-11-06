"""Cuarta version: la v3 con el saque de las ventanas largas a tiempo fijo y entregado al sacador.

Smart Tagging marca cada reanudacion con una ventana de 5 s o de 35 s. Las reglas anteriores
buscaban el arranque del balon, pero la columna de velocidad esta rota, asi que siempre caian al
centro de la ventana. En las de 35 s eso son 17,5 s, cuando el juego suele llevar 8-12 s en marcha:
los jugadores corren a ritmo de juego vivo mientras la prediccion dice D.

Cambios frente a la v3:

- Saque. En las ventanas largas la parada termina, como pronto, en inicio + 6,32 s. Es la mediana
  del primer frame con el balon a mas de 5 m/s desde el inicio de la ventana; measured_offset() la
  recalcula sin ninguna etiqueta de verdad. La dispersion por ventana es grande (P10-P90 de 2 a
  11,5 s): acierta en el centro, no en cada saque. La regla de calma sigue activa, asi que un saque
  mas lento continua en D. Las ventanas cortas no cambian: ya contienen el saque.
- Sacador. Si la ventana tiene equipo, los frames entre el saque nuevo y el centro de la ventana
  van con el peso de una prueba de control a su favor: el que saca empieza con el balon. Sin esto,
  el sacador solo tenia la ventaja de un cambio de equipo, unos 0,5 s de control del rival.
- Salidas de balon. Se enlazan con el saque igual que en la v3: adelantarlo no amplia ese enlace.

Comprobado sin verdad: el ritmo de los jugadores arranca, en mediana, a unos 2 s del saque nuevo.
Sin comprobar: el efecto en spellF1 del partido completo. Los 5 minutos publicos solo se midieron
al final, como comprobacion del pipeline: despues de medirlos no cambio ningun parametro.

Descartado: un detector de saque frame a frame. Mueve casi los mismos frames que esta constante y
ademas se dispara en mas de la mitad de los tramos de juego normal.

    python solution/possession_v4.py
    python evaluation/validate_submission.py evaluation/results/inka_analytics_v4.csv
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from possession_v1 import (  # noqa: E402
    FPS, GAME, HOME, RESTART_KICK_SPEED, ROOT, restart_frame, set_pieces)
from possession_v2 import CONTROL_WEIGHT, decode, frame_score, load_match  # noqa: E402
from possession_v3 import with_control_speed  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "inka_analytics_v4.csv")

SHORT_WINDOW = 10 * FPS     # hay ventanas de 5 s y de 35 s; las cortas ya contienen el saque
RESTART_OFFSET = 158        # frames desde el inicio de una ventana larga; ver measured_offset()


def timed_restart(start, end, ball_speed):
    """Saque a tiempo fijo en las ventanas largas; en las cortas, la regla de siempre."""
    if end - start > SHORT_WINDOW:
        return start + RESTART_OFFSET
    return restart_frame(start, end, ball_speed)


def load_match_v4(game=GAME):
    return with_control_speed(load_match(game, restart_rule=timed_restart))


def measured_offset(match, game=GAME):
    """De donde sale RESTART_OFFSET: mediana del primer balon rapido en las ventanas largas."""
    speed = match.ball[:, 2]
    offsets = []
    for start, end, _ in set_pieces(game, match.names):
        fast = np.flatnonzero(speed[start:end] > RESTART_KICK_SPEED)
        if end - start > SHORT_WINDOW and len(fast):
            offsets.append(int(fast[0]))
    return int(np.median(offsets))


def frame_score_v4(match, game=GAME, fold=None):
    """Puntuacion de la v2 con el tramo saque-centro de cada ventana larga para su sacador."""
    score, summary = frame_score(match, game, fold)
    for start, end, team in set_pieces(game, match.names):
        if team is not None and end - start > SHORT_WINDOW:
            sign = 1.0 if team == HOME else -1.0
            score[start + RESTART_OFFSET:(start + end) // 2] = sign * CONTROL_WEIGHT
    return score, summary


def predict(game=GAME):
    match = load_match_v4(game)
    print("  desfase medido {}  usado {}".format(measured_offset(match, game), RESTART_OFFSET))
    score, summary = frame_score_v4(match, game)
    print("  " + "  ".join("{} {}".format(key, value) for key, value in summary.items()))
    return decode(match, score)


def write_csv(path, labels):
    """Como write_submission, pero con salto de linea fijo: el archivo no depende del sistema."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="CSV de salida")
    args = parser.parse_args()

    result = predict()
    path = write_csv(args.out, result)
    counts = {label: int(np.sum(result == label)) for label in "HADX"}
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("escrito en {}".format(path))

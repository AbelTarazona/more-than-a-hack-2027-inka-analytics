"""Compara versiones sin mirar la verdad de los 5 minutos publicos.

Validacion cruzada con bloques alternos de 60 s: cada variante ve las pruebas de un grupo de
bloques y se examina en el otro, y luego al reves. Hay dos examenes:

    fases    fases de Smart Tagging no vistas.
    balon    frames donde solo broadcast ve el balon y alguien lo controla. Un modelo sin broadcast
             nunca los ve; a la variante con broadcast se le ocultan los de los bloques de examen.

Limitaciones: ninguno de los dos es la verdad ni estima spellF1. Ademas, el acuerdo sube cuando hay
menos cambios de equipo, asi que se lee junto al numero de posesiones y no sirve para ajustar
constantes.

    python solution/compare_versions.py
"""

import dataclasses
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from broadcast_ball import fuse_ball  # noqa: E402
from possession_v1 import control_evidence  # noqa: E402
from possession_v2 import (  # noqa: E402
    FPS, GAME, decode, frame_score, load_match, phase_frame_fold, phase_votes)
from possession_v3 import with_control_speed  # noqa: E402
from possession_v4 import frame_score_v4, load_match_v4  # noqa: E402


def agreement(labels, votes):
    """Parte de los frames con voto y juego vivo en que la etiqueta coincide con el voto."""
    alive = (labels == "H") | (labels == "A")
    right = ((labels == "H") & (votes > 0)) | ((labels == "A") & (votes < 0))
    return right.sum() / max((alive & (votes != 0)).sum(), 1)


def describe(labels):
    """Numero de posesiones, mediana en segundos y parte de las que duran menos de 2 s."""
    durations = []
    start = None
    for frame in range(len(labels) + 1):
        value = labels[frame] if frame < len(labels) else "D"
        if start is not None and value != labels[start]:
            durations.append(frame - start)
            start = None
        if start is None and value in ("H", "A"):
            start = frame
    durations = np.array(durations) / FPS
    return len(durations), float(np.median(durations)), float((durations < 2).mean())


def main():
    tactical = load_match()
    fixed = with_control_speed(tactical)
    fourth = load_match_v4()
    fused, filled = fuse_ball(fixed.players, fixed.ball, fixed.blocks)

    block = phase_frame_fold(np.arange(fixed.n_frames))
    home, away = control_evidence(fixed.players, fused, fixed.team_of_track)
    ball_votes = np.zeros(fixed.n_frames)
    ball_votes[filled & home] = 1
    ball_votes[filled & away] = -1
    print("frames de examen del balon: {}".format(int(np.count_nonzero(ball_votes))))

    def variant_v2(fold):
        return decode(tactical, frame_score(tactical, fold=fold)[0])

    def variant_v3(fold):
        return decode(fixed, frame_score(fixed, fold=fold)[0])

    def variant_v4(fold):
        return decode(fourth, frame_score_v4(fourth, fold=fold)[0])

    def variant_broadcast(fold):
        ball = fixed.ball.copy()
        visible = filled & (block == fold)
        ball[visible] = fused[visible]
        return decode(fixed, frame_score(dataclasses.replace(fixed, ball=ball), fold=fold)[0])

    print("\n{:<26}{:>8}{:>8}{:>12}{:>11}{:>8}{:>8}".format(
        "variante", "fases", "balon", "posesiones", "mediana s", "< 2 s", "D"))
    for name, variant in (("v2", variant_v2), ("v3", variant_v3), ("v4", variant_v4),
                          ("v3 + broadcast", variant_broadcast)):
        phases, balls = [], []
        for fold in (0, 1):
            labels = variant(fold)
            phases.append(agreement(
                labels, phase_votes(GAME, fixed.names, fixed.n_frames, 1 - fold)))
            balls.append(agreement(labels, np.where(block != fold, ball_votes, 0)))
        count, median, short = describe(labels)
        print("{:<26}{:>8.4f}{:>8.4f}{:>12d}{:>11.1f}{:>8.1%}{:>8d}".format(
            name, np.mean(phases), np.mean(balls), count, median, short,
            int((labels == "D").sum())))


if __name__ == "__main__":
    main()

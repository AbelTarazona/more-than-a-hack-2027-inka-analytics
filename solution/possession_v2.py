"""Segunda version: junta varias pruebas y elige la historia de posesion mas coherente.

La v1 solo aceptaba una prueba, el control con el balon visible, y el balon falta en el 41 % de
los frames puntuados. La v2 reutiliza su deteccion de juego parado y combina dos pruebas:

    1. Control directo: como en la v1.
    2. Fases de Smart Tagging: indican que equipo ataca, incluso sin balon visible. Son un indicio
       debil y automatico, asi que pesan poco y nunca se copian como etiquetas.

Cada frame recibe una puntuacion: positiva si favorece al local, negativa si favorece al
visitante. El algoritmo de Viterbi elige entonces la secuencia de equipos que suma mas puntos
pagando un coste por cada cambio. Asi un rebote no basta para cambiar, y el cambio se coloca solo.

Descartado: puntuar las cadenas de pases. Se probo y no cambio ni un frame, porque una cadena sin
toques del rival ya queda con ese equipo.

    python solution/possession_v2.py
    python evaluation/validate_submission.py evaluation/results/inka_analytics_v2.csv

Las constantes salen del razonamiento futbolistico y de CHALLENGE.md, no de probar valores contra
los 5 minutos publicos (RULES.md, EVALUATION.md). Para decidir entre versiones sin esa ventana,
ver solution/compare_versions.py.
"""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from possession_v1 import (  # noqa: E402
    AWAY, CHANGE_MIN_FRAMES, FPS, GAME, GAMES, HOME, ROOT, control_evidence, dead_ball_mask,
    half_video_blocks, load_smart_tagging, read_tracking, require, restart_frame, runs, set_pieces,
    source_frames, team_names, team_of_each_track, write_submission)

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "inka_analytics_v2.csv")

# --- Peso de cada prueba, por frame ---
CONTROL_WEIGHT = 1.0        # ver el control directo es la prueba mas fuerte
PHASE_WEIGHT = 0.1          # diez frames de fase valen uno de control: indicio, no verdad

# --- Coste de cambiar de equipo ---
# Un cambio definitivo necesita 0,5 s de control neto, igual que en la v1. Un episodio aislado del
# rival paga ida y vuelta, asi que necesita 1 s: el reto pide conservar el balon, no tocarlo.
SWITCH_COST = CHANGE_MIN_FRAMES

# Fases de Smart Tagging (OVERVIEW.md). Su etiqueta Team es el equipo que hace la accion:
# en las de ataque tiene el balon; en las de defensa lo tiene el otro.
ATTACKING_PHASES = {
    "build up", "progression", "creating chances", "attacking transition", "long ball",
    "ball in final third", "ball in the box", "players in the box"}
DEFENDING_PHASES = {
    "defending in attacking third", "defending in middle third",
    "defending in defensive third", "defensive transition"}


@dataclass
class Match:
    """Lo que no depende de la version: tracking, pruebas directas y juego parado."""
    n_frames: int
    names: dict
    blocks: list
    players: np.ndarray
    ball: np.ndarray
    team_of_track: np.ndarray
    dead: np.ndarray
    restarts: dict


def load_match(game=GAME, restart_rule=restart_frame):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    team_of_track = team_of_each_track(meta_path)
    names = team_names(meta_path)
    blocks = half_video_blocks(game)
    players, ball = read_tracking(raw_path, n_frames, len(team_of_track), pitch)
    pieces = set_pieces(game, names)
    dead, restarts, _ = dead_ball_mask(players, ball, pieces, pitch, blocks, game, restart_rule)
    return Match(n_frames, names, blocks, players, ball, team_of_track, dead, restarts)


def phase_frame_fold(start_frame):
    """Divide el partido en bloques de 60 s alternos, para compare_versions.py."""
    return (start_frame // (60 * FPS)) % 2


def phase_votes(game, names, n_frames, fold=None):
    """Votos por frame: +1 si las fases dicen local, -1 si visitante, 0 si no dicen o discrepan.

    `fold` usa solo las fases de un bloque par (0) o impar (1); None las usa todas.
    """
    teams = {names[HOME].strip().lower(): HOME, names[AWAY].strip().lower(): AWAY}
    change = np.zeros(n_frames + 1, dtype=np.int32)
    for event in load_smart_tagging(game):
        code = (event["code"] or "").strip().lower()
        team = teams.get((event["tags"].get("Team") or "").strip().lower())
        if team is None or (code not in ATTACKING_PHASES and code not in DEFENDING_PHASES):
            continue
        if fold is not None and phase_frame_fold(event["start_frame"]) != fold:
            continue
        in_possession = team if code in ATTACKING_PHASES else 1 - team
        sign = 1 if in_possession == HOME else -1
        start = max(0, event["start_frame"])
        end = min(n_frames, event["end_frame"])
        if end > start:
            change[start] += sign
            change[end] -= sign
    return np.sign(np.cumsum(change[:-1]))


def viterbi(score, initial=None, tie_break=None):
    """Mejor secuencia de equipos dado `score` (positivo = local).

    El local suma `score` por frame y cambiar cuesta SWITCH_COST. En un empate se prefiere cambiar
    lo mas tarde posible: sin pruebas la posesion sigue con el ultimo equipo, como pide el reto.

    `initial` es el equipo que reanuda: empezar con el otro cuesta un cambio. `tie_break` decide
    solo cuando no hay ninguna prueba en todo el tramo.
    """
    value = {HOME: 0.0, AWAY: 0.0}
    if initial is not None:
        value[1 - initial] = -SWITCH_COST
    came_from_other = np.zeros((len(score), 2), dtype=bool)
    values = score.tolist()
    for frame in range(1, len(score)):
        home, away = value[HOME] + values[frame - 1], value[AWAY]
        came_from_other[frame, HOME] = away - SWITCH_COST >= home
        came_from_other[frame, AWAY] = home - SWITCH_COST >= away
        value[HOME] = max(home, away - SWITCH_COST)
        value[AWAY] = max(away, home - SWITCH_COST)
    value[HOME] += values[-1]

    if value[HOME] != value[AWAY]:
        state = HOME if value[HOME] > value[AWAY] else AWAY
    elif tie_break is not None:
        state = tie_break
    else:
        state = HOME

    path = np.empty(len(score), dtype=np.int8)
    for frame in range(len(score) - 1, -1, -1):
        path[frame] = state
        if came_from_other[frame, state]:
            state = 1 - state
    return path


def frame_score(match, game=GAME, fold=None, use_phases=True):
    """Puntuacion por frame. Devuelve (puntuacion, resumen de cuantas pruebas hay)."""
    home, away = control_evidence(match.players, match.ball, match.team_of_track)
    control = home.astype(np.float64) - away
    if use_phases:
        phases = phase_votes(game, match.names, match.n_frames, fold).astype(np.float64)
    else:
        phases = np.zeros(match.n_frames)

    score = CONTROL_WEIGHT * control + PHASE_WEIGHT * phases
    summary = {"frames_control": int(np.count_nonzero(control)),
               "frames_fase": int(np.count_nonzero(phases))}
    return score, summary


def decode(match, score):
    """Etiquetas H/A/D/X del partido completo a partir de la puntuacion por frame."""
    labels = np.full(match.n_frames, "X", dtype="<U1")  # fuera de las dos partes no se puntua
    restart_frames = sorted(match.restarts)
    for _, first, last in match.blocks:
        block_end = last + 1
        block_dead = np.zeros(match.n_frames, dtype=bool)
        block_dead[first:block_end] = match.dead[first:block_end]
        labels[block_dead] = "D"

        in_block = np.zeros(match.n_frames, dtype=bool)
        in_block[first:block_end] = ~match.dead[first:block_end]
        previous = None
        for start, end in runs(in_block):
            near = [frame for frame in restart_frames if abs(frame - start) <= FPS]
            initial = match.restarts[min(near, key=lambda f: abs(f - start))] if near else None
            path = viterbi(score[start:end], initial, previous)
            previous = int(path[-1])
            labels[start:end] = np.where(path == HOME, "H", "A")
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="CSV de salida")
    args = parser.parse_args()

    print("leyendo tracking ...")
    match = load_match()
    score, summary = frame_score(match)
    print("  " + "  ".join("{} {}".format(key, value) for key, value in summary.items()))
    result = decode(match, score)
    path = write_submission(args.out, result)
    counts = {label: int(np.sum(result == label)) for label in "HADX"}
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("escrito en {}".format(path))

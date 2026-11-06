"""Primera version propia: dos preguntas por frame, en este orden.

    1. Esta parado el juego?  -> D
    2. Si no, quien controla el balon?  -> H o A

La pregunta 2 es una maquina de estados: el equipo con posesion la conserva hasta que el rival
demuestra CONTROL, no un simple toque. Cuando se confirma un cambio, la etiqueta se corrige hacia
atras hasta el momento en que empezo, porque spellF1 castiga llegar tarde a los cambios.

    python solution/possession_v1.py
    python evaluation/validate_submission.py evaluation/results/inka_analytics_v1.csv

Regla importante del reto: las constantes de abajo salen del razonamiento futbolistico, NO de
probar valores contra los 5 minutos publicos. Esa ventana sirve para comprobar que el pipeline
funciona, no para elegir parametros (RULES.md, EVALUATION.md).

Pendiente para versiones siguientes: usar las fases de Smart Tagging con equipo para resolver los
huecos largos del balon, y detectar posesion por cadenas de pases.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import team_names, team_of_each_track, write_submission  # noqa: E402
from games import (  # noqa: E402
    GAME, GAMES, anchor_video_frame, half_video_blocks, require, source_frames)
from load_smart_tagging import load as load_smart_tagging  # noqa: E402

FPS = GAMES[GAME]["fps"]
DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "inka_analytics_v1.csv")

HOME, AWAY = 0, 1

# --- Control del balon ---
CONTROL_RADIUS_M = 2.0      # un jugador que controla tiene el balon a un par de metros
CONTROL_MARGIN_M = 0.5      # el rival debe estar claramente mas lejos; si no, es disputa
MAX_CONTROL_SPEED = 10.0    # m/s; a mas velocidad el balon viaja, no esta controlado

# --- Cambio de posesion ---
CHANGE_WINDOW = 2 * FPS     # se mira lo ocurrido en los ultimos 2 s
CHANGE_MIN_FRAMES = FPS // 2  # el rival necesita 0,5 s de control; un rebote dura menos
CHANGE_DOMINANCE = 2.0      # y al menos el doble de control que el equipo actual

# --- Juego parado ---
OUT_MARGIN_M = 1.0          # margen por el ruido de posicion cerca de las lineas
OUT_MIN_FRAMES = FPS // 2   # 0,5 s fuera, para no creer un punto suelto del tracking
OUT_TO_RESTART_MAX = 40 * FPS  # una reanudacion rara vez tarda mas de 40 s
WALKING_SPEED = 1.3         # m/s; por debajo, los jugadores caminan o estan quietos
ACTIVITY_WINDOW = 2 * FPS   # la velocidad media se promedia en 2 s
LOW_ACTIVITY_MIN = 4 * FPS  # 4 s seguidos a ritmo de paseo y sin pases: juego parado
FAST_BALL_SPEED = 8.0       # m/s; un balon asi indica que se esta jugando
FAST_BALL_MARGIN = FPS      # ese balon "protege" 1 s antes y despues
RESTART_KICK_SPEED = 5.0    # m/s; el balon arranca al reanudar
MIN_ALIVE = 2 * FPS         # entre dos paradas no cabe juego de menos de 2 s
MIN_DEAD = 3 * FPS // 2     # y una parada real dura al menos 1,5 s


def read_tracking(raw_path, n_frames, n_slots, pitch):
    """Lee el tracking crudo a matrices. Posiciones en metros, NaN si falta el dato.

    Formato de cada linea: "<frame>:<jugador>;<jugador>;...:<balon>", cada grupo "x,y,velocidad".
    """
    length, width = pitch
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # el tracking cuenta desde 1; la entrega, desde 0
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] != "N":
                    values = group.split(",")
                    players[frame, slot] = (float(values[0]) * length,
                                            float(values[1]) * width, float(values[2]))
            if parts[2][0] != "N":
                values = parts[2].split(",")
                ball[frame] = (float(values[0]) * length, float(values[1]) * width,
                               float(values[2]))
    return players, ball


def rolling_mean(values, window):
    """Media movil centrada que ignora los NaN. Devuelve NaN donde no hay ningun dato."""
    valid = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    low = np.maximum(0, index - window // 2)
    high = np.minimum(len(values), index + window // 2 + 1)
    seen = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(seen > 0, (total[high] - total[low]) / seen, np.nan)


def dilate(mask, margin):
    """Marca tambien los frames a `margin` o menos de cualquier frame marcado."""
    count = np.concatenate(([0], np.cumsum(mask)))
    index = np.arange(len(mask))
    low = np.maximum(0, index - margin)
    high = np.minimum(len(mask), index + margin + 1)
    return (count[high] - count[low]) > 0


def runs(mask):
    """Tramos seguidos de True como lista de (inicio, fin), con fin exclusivo."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def control_evidence(players, ball, team_of_track):
    """Dos mascaras por frame: el local controla / el visitante controla.

    Si no se ve el balon no hay evidencia para nadie: la posesion se mantiene por defecto.
    """
    distance = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    distance = np.where(np.isnan(distance), np.inf, distance)
    nearest_home = distance[:, team_of_track == HOME].min(axis=1)
    nearest_away = distance[:, team_of_track == AWAY].min(axis=1)

    # Una comparacion con NaN da False, asi que "velocidad desconocida" no anula el control.
    ball_is_travelling = ball[:, 2] > MAX_CONTROL_SPEED
    usable = ~np.isnan(ball[:, 0]) & ~ball_is_travelling
    home = usable & (nearest_home <= CONTROL_RADIUS_M) & (
        nearest_home + CONTROL_MARGIN_M <= nearest_away)
    away = usable & (nearest_away <= CONTROL_RADIUS_M) & (
        nearest_away + CONTROL_MARGIN_M <= nearest_home)
    return home, away


def set_pieces(game, names):
    """Reanudaciones de Smart Tagging: (inicio, fin, equipo o None).

    Los saques de banda vienen con Team N/A, de ahi el None: el equipo se deduce despues.
    """
    teams = {names[HOME].strip().lower(): HOME, names[AWAY].strip().lower(): AWAY}
    pieces = []
    for event in load_smart_tagging(game):
        if (event["code"] or "").strip().lower() == "set pieces":
            team = (event["tags"].get("Team") or "").strip().lower()
            pieces.append((event["start_frame"], event["end_frame"], teams.get(team)))
    return sorted(pieces, key=lambda piece: piece[0])


def restart_frame(start, end, ball_speed):
    """Momento de la reanudacion: el balon arranca dentro del intervalo etiquetado.

    Se ignora el primer cuarto porque suele ser el balon llevado a mano hasta el lugar del saque.
    Si no se ve el balon, se usa el centro: la etiqueta marca una ventana alrededor del saque.
    """
    search_from = start + (end - start) // 4
    kicks = np.flatnonzero(ball_speed[search_from:end] > RESTART_KICK_SPEED)
    return search_from + int(kicks[0]) if len(kicks) else (start + end) // 2


def dead_ball_mask(players, ball, pieces, pitch, blocks, game, restart_rule=restart_frame):
    """True donde el juego esta parado. Devuelve ademas {frame_de_reanudacion: equipo}.

    `restart_rule(inicio, fin, velocidad_balon)` decide el frame de cada reanudacion.
    """
    n_frames = len(ball)
    length, width = pitch
    dead = np.zeros(n_frames, dtype=bool)

    # 1. Antes del saque inicial de cada parte no se juega. Los saques estan publicados en games.py.
    entry = GAMES[game]
    kickoffs = {1: anchor_video_frame(entry["kickoff"]) - 1,
                2: anchor_video_frame(entry["kickoff_2nd"]) - 1}
    for half, first, _ in blocks:
        dead[first:max(first, kickoffs[half])] = True

    # 2. Ritmo de paseo sostenido y ningun balon rapido cerca.
    speeds = players[:, :, 2]
    tracked = (~np.isnan(speeds)).sum(axis=1)
    mean_speed = np.where(tracked > 0, np.nansum(speeds, axis=1) / np.maximum(tracked, 1), np.nan)
    calm = rolling_mean(mean_speed, ACTIVITY_WINDOW) < WALKING_SPEED
    calm &= ~dilate(ball[:, 2] > FAST_BALL_SPEED, FAST_BALL_MARGIN)
    for start, end in runs(calm):
        if end - start >= LOW_ACTIVITY_MIN:
            dead[start:end] = True

    # 3. Reanudaciones de Smart Tagging: parado hasta el saque, y hacia atras mientras haya calma.
    restarts = {}
    ends_at = {}  # {frame_de_siempre: reanudacion}. Asi otra regla no cambia que salidas se enlazan
    for start, end, team in pieces:
        restart = restart_rule(start, end, ball[:, 2])
        restarts[restart] = team
        ends_at[restart_frame(start, end, ball[:, 2])] = restart
        dead[start:restart] = True
        earliest = max(0, start - OUT_TO_RESTART_MAX)
        before = start
        while before > earliest and calm[before - 1]:
            before -= 1
        dead[before:start] = True

    # 4. Balon fuera del campo: parado hasta la siguiente reanudacion, si llega pronto.
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        outside = ((x < -OUT_MARGIN_M) | (x > length + OUT_MARGIN_M)
                   | (y < -OUT_MARGIN_M) | (y > width + OUT_MARGIN_M))
    restart_frames = np.array(sorted(ends_at), dtype=np.int64)
    n_out = 0
    for start, end in runs(outside):
        if end - start < OUT_MIN_FRAMES:
            continue
        n_out += 1
        position = np.searchsorted(restart_frames, start)
        following = restart_frames[position] if position < len(restart_frames) else None
        if following is not None and following - start <= OUT_TO_RESTART_MAX:
            stop = ends_at[following]
            dead[start:stop if stop > start else end] = True
        else:
            dead[start:end] = True

    # 5. Limpieza: quitar tramos imposibles de juego y de parada.
    for start, end in runs(~dead):
        if end - start < MIN_ALIVE:
            dead[start:end] = True
    for start, end in runs(dead):
        if end - start < MIN_DEAD:
            dead[start:end] = False
    return dead, restarts, n_out


def label_alive_run(start, end, evidence, initial):
    """Etiqueta un tramo de juego con la maquina de estados. Devuelve (etiquetas, equipo final).

    `initial` es el equipo que reanuda, o None si no se sabe. En ese caso, gana el primero que
    demuestre control, y el inicio del tramo se le asigna a el.
    """
    count = {team: np.concatenate(([0], np.cumsum(evidence[team][start:end])))
             for team in (HOME, AWAY)}
    length = end - start
    state = initial
    labels = np.full(length, -1, dtype=np.int8)
    current_from = 0

    for frame in range(length):
        low = max(0, frame - CHANGE_WINDOW + 1)
        for challenger in (HOME, AWAY):
            if challenger == state:
                continue
            ours = 0 if state is None else count[state][frame + 1] - count[state][low]
            theirs = count[challenger][frame + 1] - count[challenger][low]
            if theirs < CHANGE_MIN_FRAMES or theirs < CHANGE_DOMINANCE * ours:
                continue

            # Cambio confirmado. Se fecha cuando EMPEZO: el primer control del rival en la
            # ventana posterior al ultimo control nuestro.
            window = evidence[challenger][start + low:start + frame + 1]
            onset = low + int(np.argmax(window))
            if state is not None:
                ours_window = evidence[state][start + low:start + frame + 1]
                if ours_window.any():
                    last_ours = low + len(ours_window) - 1 - int(np.argmax(ours_window[::-1]))
                    later = window[last_ours - low + 1:]
                    if later.any():
                        onset = last_ours + 1 + int(np.argmax(later))
                labels[current_from:onset] = state
                current_from = max(current_from, onset)
            state = challenger
            break

    if state is None:
        # Sin evidencia en todo el tramo: que lo decida quien lo llame.
        return labels, None
    labels[current_from:] = state
    return labels, state


def predict(game=GAME):
    """Etiquetas H/A/D/X del partido completo."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    team_of_track = team_of_each_track(meta_path)
    names = team_names(meta_path)
    blocks = half_video_blocks(game)

    print("leyendo tracking ...")
    players, ball = read_tracking(raw_path, n_frames, len(team_of_track), pitch)
    home_evidence, away_evidence = control_evidence(players, ball, team_of_track)
    evidence = {HOME: home_evidence, AWAY: away_evidence}
    pieces = set_pieces(game, names)
    dead, restarts, n_out = dead_ball_mask(players, ball, pieces, pitch, blocks, game)

    labels = np.full(n_frames, "X", dtype="<U1")  # fuera de las dos partes no se puntua
    letters = {HOME: "H", AWAY: "A"}
    n_alive_runs = 0
    for _, first, last in blocks:
        block_end = last + 1
        block_dead = np.zeros(n_frames, dtype=bool)
        block_dead[first:block_end] = dead[first:block_end]
        labels[block_dead] = "D"

        in_block = np.zeros(n_frames, dtype=bool)
        in_block[first:block_end] = ~dead[first:block_end]
        previous = None
        for start, end in runs(in_block):
            n_alive_runs += 1
            # Reanudacion mas cercana al inicio del tramo, si la hay.
            near = [frame for frame in restarts if abs(frame - start) <= FPS]
            initial = restarts[min(near, key=lambda f: abs(f - start))] if near else None
            run_labels, state = label_alive_run(start, end, evidence, initial)
            if state is None:
                # Tramo sin balon visible: se mantiene el ultimo equipo conocido.
                state = previous if previous is not None else HOME
                run_labels[:] = state
            previous = state
            labels[start:end] = np.where(run_labels == HOME, letters[HOME], letters[AWAY])

    print("  reanudaciones Smart Tagging {}  salidas de balon detectadas {}  tramos de juego {}"
          .format(len(pieces), n_out, n_alive_runs))
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="CSV de salida")
    args = parser.parse_args()

    result = predict()
    path = write_submission(args.out, result)
    counts = {label: int(np.sum(result == label)) for label in "HADX"}
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("escrito en {}".format(path))

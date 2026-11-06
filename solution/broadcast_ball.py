"""Rellena los huecos del balon tactico con el tracking de la realizacion (broadcast).

Broadcast no esta sincronizado con el video tactico y no hay tabla de correspondencia. Aqui la
sincronia se calcula con un algoritmo y nunca se escribe a mano:

    1. Por cada parte, se busca el desfase de frames con que mas coinciden los dos balones.
    2. Se afina frame a frame con el error mediano mientras el balon se mueve.
    3. Solo se acepta un frame de broadcast si sus jugadores coinciden con los del tactico en ese
       mismo instante. Asi se descartan las repeticiones, que muestran otro momento del partido.

    python solution/broadcast_ball.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from possession_v1 import (  # noqa: E402
    GAME, GAMES, read_tracking, source_frames, team_of_each_track)
from games import asset_paths  # noqa: E402

SAMPLE_EVERY = 40           # para la busqueda gruesa basta un frame de cada 40
COARSE_STEP = 5             # 0,2 s: en ese tiempo el balon apenas se aleja del radio de abajo
MATCH_RADIUS_M = 2.0        # dos balones a menos de 2 m cuentan como el mismo
REFINE_RANGE = 2 * COARSE_STEP
MOVING_BALL_SPEED = 3.0     # m/s; con el balon quieto cualquier desfase parece bueno

SPEED_SPAN = 2              # velocidad medida entre 2 frames antes y 2 despues, para filtrar ruido

GATE_MAX_PLAYER_ERROR_M = 3.0  # error mediano por encima del cual los jugadores no coinciden
GATE_MIN_PLAYERS = 4        # con menos jugadores en pantalla no se puede comprobar
GATE_CHUNK = 5000           # frames por lote, para no llenar la memoria


def with_derived_speed(ball, fps):
    """Copia del balon con la velocidad calculada a partir de sus posiciones.

    La columna de velocidad del tracking falta en mas del 90 % de los frames con balon visible y
    nunca supera 7,3 m/s, asi que no sirve para detectar pases.
    """
    result = ball.copy()
    result[:, 2] = np.nan
    span = 2 * SPEED_SPAN
    moved = np.hypot(ball[span:, 0] - ball[:-span, 0], ball[span:, 1] - ball[:-span, 1])
    result[SPEED_SPAN:-SPEED_SPAN, 2] = moved * fps / span
    return result


def read_broadcast(game=GAME):
    """Jugadores y balon de broadcast, en metros, sobre su propia linea temporal."""
    paths = asset_paths(source="broadcast")
    team_of_track = team_of_each_track(paths["atd_meta"])
    players, ball = read_tracking(paths["atd_raw"], source_frames("broadcast", game),
                                  len(team_of_track), GAMES[game]["pitch"])
    return players, with_derived_speed(ball, GAMES[game]["fps"])


def coarse_offset(tactical, broadcast, first, last):
    """Desfase tactico - broadcast con mas balones coincidentes en esta parte."""
    frames = np.arange(first, last + 1, SAMPLE_EVERY)
    frames = frames[~np.isnan(tactical[frames, 0])]
    n_broadcast = len(broadcast)
    offsets = np.arange(-n_broadcast, last + 1, COARSE_STEP)
    best_hits, best_offset = -1, 0
    for batch in np.array_split(offsets, 200):
        index = frames[:, None] - batch[None, :]
        inside = (index >= 0) & (index < n_broadcast)
        clipped = np.clip(index, 0, n_broadcast - 1)
        distance = np.hypot(tactical[frames, 0, None] - broadcast[clipped, 0],
                            tactical[frames, 1, None] - broadcast[clipped, 1])
        hits = (inside & (distance < MATCH_RADIUS_M)).sum(axis=0)
        where = int(hits.argmax())
        if hits[where] > best_hits:
            best_hits, best_offset = int(hits[where]), int(batch[where])
    return best_offset, best_hits / max(len(frames), 1)


def refine_offset(tactical, broadcast, first, last, start):
    """Afina el desfase: menor error mediano con el balon en movimiento."""
    best_error, best_offset = np.inf, start
    for offset in range(start - REFINE_RANGE, start + REFINE_RANGE + 1):
        frames = np.arange(max(first, offset), min(last + 1, len(broadcast) + offset))
        other = broadcast[frames - offset]
        moving = tactical[frames, 2] > MOVING_BALL_SPEED
        distance = np.hypot(tactical[frames, 0] - other[:, 0], tactical[frames, 1] - other[:, 1])
        distance = distance[moving & ~np.isnan(distance)]
        error = float(np.median(distance)) if len(distance) else np.inf
        if error < best_error:
            best_error, best_offset = error, offset
    return best_offset, best_error


def player_error(tactical_players, broadcast_players):
    """Por frame: error mediano de cada jugador de broadcast al tactico mas cercano."""
    n_frames = len(tactical_players)
    errors = np.full(n_frames, np.nan, dtype=np.float32)
    seen = np.zeros(n_frames, dtype=np.int16)
    for low in range(0, n_frames, GATE_CHUNK):
        high = min(n_frames, low + GATE_CHUNK)
        ours = tactical_players[low:high, None, :, :2]
        theirs = broadcast_players[low:high, :, None, :2]
        distance = np.hypot(ours[..., 0] - theirs[..., 0], ours[..., 1] - theirs[..., 1])
        nearest = np.where(np.isnan(distance), np.inf, distance).min(axis=2)
        nearest[~np.isfinite(nearest)] = np.nan
        seen[low:high] = (~np.isnan(nearest)).sum(axis=1)
        valid = seen[low:high] > 0
        errors[low:high][valid] = np.nanmedian(nearest[valid], axis=1)
    return errors, seen


def fuse_ball(players, ball, blocks, game=GAME, verbose=True):
    """Devuelve el balon tactico con huecos rellenados y la mascara de frames rellenados."""
    broadcast_players, broadcast_ball = read_broadcast(game)
    fused = ball.copy()
    filled = np.zeros(len(ball), dtype=bool)

    for half, first, last in blocks:
        coarse, hit_rate = coarse_offset(ball, broadcast_ball, first, last)
        offset, error = refine_offset(ball, broadcast_ball, first, last, coarse)

        frames = np.arange(max(first, offset), min(last + 1, len(broadcast_ball) + offset))
        errors, seen = player_error(players[frames], broadcast_players[frames - offset])
        trusted = (errors <= GATE_MAX_PLAYER_ERROR_M) & (seen >= GATE_MIN_PLAYERS)

        other = broadcast_ball[frames - offset]
        usable = trusted & np.isnan(ball[frames, 0]) & ~np.isnan(other[:, 0])
        fused[frames[usable]] = other[usable]
        filled[frames[usable]] = True

        both = trusted & ~np.isnan(ball[frames, 0]) & ~np.isnan(other[:, 0])
        gap = np.hypot(ball[frames, 0] - other[:, 0], ball[frames, 1] - other[:, 1])[both]
        if verbose:
            n_block = last - first + 1
            print("parte {}: desfase {}  coincidencia {:.0%}  error {:.2f} m".format(
                half, offset, hit_rate, error))
            print("  frames de broadcast fiables {:.1%}  distancia entre balones p50 {:.2f} m"
                  "  p90 {:.2f} m".format(trusted.sum() / n_block, np.median(gap),
                                          np.percentile(gap, 90)))
            print("  balon visible {:.1%} -> {:.1%}".format(
                (~np.isnan(ball[first:last + 1, 0])).mean(),
                (~np.isnan(fused[first:last + 1, 0])).mean()))
    return fused, filled


if __name__ == "__main__":
    from possession_v2 import load_match
    match = load_match()
    fuse_ball(match.players, with_derived_speed(match.ball, GAMES[GAME]["fps"]), match.blocks)

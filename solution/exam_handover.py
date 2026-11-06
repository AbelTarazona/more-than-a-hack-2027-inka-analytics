"""Examen de cambios de posesion sin usar la verdad publica.

Los examenes de compare_versions.py miden equipos, no cambios. Este simula el problema real: en
minutos donde el balon se ve bien se le ponen encima los huecos reales de otro momento del partido,
y se pide recuperar lo que se habria decidido viendolo. Asi la respuesta sale de la propia
evidencia del tracking, sin etiquetas humanas (RULES.md admite pseudo-etiquetas).

    verdad     v3 solo con control directo y todo el balon disponible, sin fases.
    examen     bloques de 60 s con buena cobertura real, fuera de los 5 minutos publicos.
    entrada    el mismo partido con el balon borrado en esos bloques segun el patron real de huecos
               de otro minuto. Un metodo tampoco ve las fases de esos bloques.
    nota       spellF1 y transF1 oficiales contra esa verdad, promediados en dos pliegues y tres
               desplazamientos del patron.

Limitaciones: la verdad es una pseudo-verdad, asi que esta nota no estima la nota real, y hereda
los errores del control directo. Sirve para comparar metodos de rellenar huecos, no para fijar el
coste de cambio, porque tambien premia lo que se parezca a la v3.

Un metodo es una funcion `metodo(match, fold) -> etiquetas` que no debe mirar nada mas que `match`
y las fases del pliegue `fold` (`phase_votes(GAME, names, n_frames, fold)`).

    python solution/exam_handover.py
"""

import dataclasses
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession_v1 import ROOT  # noqa: E402,F401
from possession_v2 import FPS, decode, frame_score, load_match, phase_frame_fold  # noqa: E402
from possession_v3 import with_control_speed  # noqa: E402

PUBLIC_WINDOW = (11185, 18685)      # nunca entra en la nota
MIN_COVERAGE = 0.7                  # bloque valido si el balon real se ve en el 70 % del juego vivo
SHIFT_BLOCKS = (3, 7, 11)           # el patron de huecos viene de 3, 7 u 11 minutos despues


def load(cache=None):
    """Partido v3. `cache` guarda las matrices en local para no releer el tracking."""
    if cache and os.path.exists(cache):
        with open(cache, "rb") as handle:
            return pickle.load(handle)
    match = with_control_speed(load_match())
    if cache:
        with open(cache, "wb") as handle:
            pickle.dump(match, handle)
    return match


def graded_mask(match):
    mask = np.zeros(match.n_frames, dtype=bool)
    for _, first, last in match.blocks:
        mask[first:last + 1] = True
    return mask


def exam_blocks(match):
    """Mascara de frames de examen: bloques de 60 s con buena cobertura y fuera de lo publico."""
    alive = graded_mask(match) & ~match.dead
    seen = ~np.isnan(match.ball[:, 0])
    block_id = np.arange(match.n_frames) // (60 * FPS)
    allowed = np.zeros(match.n_frames, dtype=bool)
    chosen = 0
    for block in np.unique(block_id):
        frames = block_id == block
        n_alive = int((frames & alive).sum())
        public = (frames & (np.arange(match.n_frames) >= PUBLIC_WINDOW[0])
                  & (np.arange(match.n_frames) < PUBLIC_WINDOW[1])).any()
        if (n_alive >= 30 * FPS and not public
                and (frames & alive & seen).sum() / n_alive >= MIN_COVERAGE):
            allowed[frames] = True
            chosen += 1
    return allowed & graded_mask(match), chosen


def pseudo_truth(match):
    return decode(match, frame_score(match, use_phases=False)[0])


def blackout(match, exam_frames, shift_blocks):
    """Copia del partido con los huecos reales de otro minuto sobre los frames de examen."""
    hidden = np.roll(np.isnan(match.ball[:, 0]), shift_blocks * 60 * FPS) & exam_frames
    ball = match.ball.copy()
    ball[hidden] = np.nan
    return dataclasses.replace(match, ball=ball)


def score(truth, labels, exam_frames):
    gt = pm.label_codes(np.where(exam_frames, truth, "X"))
    pred = pm.label_codes(np.where(exam_frames, labels, "X"))
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gt_trans, pred_trans = pm.transitions(gt, runs), pm.transitions(filled, runs)
    live = exam_frames & np.isin(truth, ("H", "A"))
    return {"spell": pm.spell_scores(gt, filled, runs)["mean_f1"],
            "trans12": pm.transition_scores(gt_trans, pred_trans, 12)["f1"],
            "trans25": pm.transition_scores(gt_trans, pred_trans, 25)["f1"],
            "equipo": float((labels[live] == truth[live]).mean()),
            "n_pred": len(pm.possession_spells(filled, runs)),
            "n_gt": len(pm.possession_spells(gt, runs))}


def run_exam(method, match=None, cache=None):
    """Nota media de un metodo y una lista de notas por pliegue y desplazamiento."""
    match = load(cache) if match is None else match
    truth = pseudo_truth(match)
    allowed, _ = exam_blocks(match)
    block = phase_frame_fold(np.arange(match.n_frames))
    results = []
    for fold in (0, 1):
        exam_frames = allowed & (block != fold)
        for shift in SHIFT_BLOCKS:
            labels = method(blackout(match, exam_frames, shift), fold)
            results.append(score(truth, labels, exam_frames))
    mean = {key: float(np.mean([r[key] for r in results])) for key in results[0]}
    return mean, results


def v3_method(match, fold):
    return decode(match, frame_score(match, fold=fold)[0])


def main():
    cache = os.environ.get("EXAM_CACHE")
    match = load(cache)
    allowed, chosen = exam_blocks(match)
    truth = pseudo_truth(match)
    print("bloques de examen {}  frames {}  posesiones verdad {}".format(
        chosen, int(allowed.sum()), score(truth, truth, allowed)["n_gt"]))
    mean, results = run_exam(v3_method, match)
    print("v3: " + "  ".join("{} {:.4f}".format(k, v) for k, v in mean.items()))
    print("spell por corrida: " + " ".join("{:.3f}".format(r["spell"]) for r in results))


if __name__ == "__main__":
    main()

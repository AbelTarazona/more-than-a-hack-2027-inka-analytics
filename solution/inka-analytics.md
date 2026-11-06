# Who has the ball? Rule-based possession from tracking and Smart Tagging

*Technical document, More than a Hack 2027, Phase 1. Match: BAR-ATH.*

## Summary

We label each frame in two steps. First we decide whether the ball is dead. If it is alive, a
Viterbi decoder picks the most coherent sequence of teams, given evidence of ball control and
Smart Tagging phases. The submitted pipeline trains nothing and uses no labels, so each of its
decisions can be traced to a rule in the code.

We chose this approach because `spellF1` counts possessions as objects. The cost of a false change
is therefore explicit in the decoder, and a brief touch by the opponent does not start a possession.

## Approach

### Step 1: dead ball

A frame is dead if any of these rules marks it:

- **Before kickoff.** Frames before each half's published kickoff.
- **Sustained calm.** Mean player speed under 1.3 m/s (2 s average) for at least 4 s. The rule also
  has a guard for a ball above 8 m/s, but it reads the provided speed column, which never reaches
  that value here. In practice, player speed alone decides.
- **Set pieces.** From the start of a Smart Tagging set-piece window to the restart, and backwards
  while players stay under the calm speed, up to 40 s. This backward step has no 4 s minimum.
- **Ball out.** The ball is more than 1 m outside the pitch for about 0.5 s. It stays dead until
  the next restart if that comes within 40 s. In long windows, the 40 s test uses the window's middle,
  as in v3, while the dead stretch ends at the v4 restart.

Live stretches under 2 s then become dead, and dead stretches under about 1.5 s become live.

### Step 2: which team

Each live frame gets a score, positive for home and negative for away:

- **Direct control, weight 1.** The ball is visible, the nearest player is within 2 m, the nearest
  opponent is at least 0.5 m farther away, and the derived ball speed is not above 10 m/s. Frames
  where that speed cannot be computed pass this last test.
- **Smart Tagging phases, weight 0.1.** Attacking phases vote for the tagged team and defending
  phases for its opponent. Ten frames of phase evidence equal one frame of control.
- **Set-piece taker, weight 1.** In long windows only, this score replaces the two above from the
  restart to the middle of the window.

Viterbi then runs once per live stretch. It finds the team sequence with the highest total score,
paying 12 points for each change. That cost is about half a second of net control. An isolated
opponent episode pays it twice, so it needs about a second. The value came from reasoning about the
rule, not from a search.

A stretch can start from a known team. If a restart with a named team lies within 1 s of the
stretch's first frame, starting with the other team costs one change. Otherwise, the stretch starts
neutral: the first net evidence claims it from its first frame, and the previous team only breaks
exact ties. A change of team across a stoppage is therefore free.

### How the versions evolved

| version | change |
|---|---|
| v1 | Dead-ball rules and a state machine driven only by direct control |
| v2 | Adds Smart Tagging phases and replaces the state machine with Viterbi |
| v3 | Derives ball speed from positions, because the provided column is unusable |
| v4 | Fixes the restart time in long set-piece windows and gives the taker the ball |

The v4 change addresses a measured problem. Smart Tagging set-piece windows last either 5 s or
35 s. Earlier versions fell back to the middle of the window, 17.5 s into the long ones, by which
time play had usually been running for 8–12 s. v4 ends the stoppage no earlier than 6.32 s (158
frames) after the window starts. Compared with v3, this turned 11,697 dead frames into live play and
only 28 the other way, and 308 frames changed team.

## Possession rules

- **Loose balls.** Within a live stretch, possession stays with the current team until evidence
  overcomes the change cost. On an exact tie, Viterbi places the change as late as possible.
- **Challenges.** If both teams are within the 0.5 m margin, neither gets evidence. Within a stretch
  whose team is already established, a rebound or a touch shorter than the change cost changes
  nothing.
- **Passes.** A ball above 10 m/s is travelling, so a pass near an opponent is not control.
- **Restarts.** In short windows, the restart is the first frame above 5 m/s in the provided speed
  column after the first quarter of the window. That column is often empty, so this falls back to
  the middle of the window. In long windows, it is the window start plus 158 frames, unless the calm
  rule keeps the frames dead.
- **Throw-ins.** Smart Tagging gives no team, so the evidence decides.
- **Outside the halves.** Labelled `X`. The file contains no `X` on graded frames.

## Assets

| asset | use |
|---|---|
| Tactical tracking and metadata | Used: positions, player speeds, team of each track |
| Tactical Smart Tagging | Used: set-piece windows and phases, as evidence and never copied as labels |
| Published kickoff anchors (`games.py`) | Used: start of play in each half |
| Provided ball-speed column | Used only for short-window restarts and the inactive calm guard. Replaced by speed derived from positions for control and for measuring `RESTART_OFFSET`, because it is missing on most frames and never exceeds 7.3 m/s |
| Broadcast tracking | Tried and dropped from the prediction. Kept as an independent exam |
| Video and homography | Not used |

## Training data

None. The submitted pipeline uses no labelled dataset, from this game or any other, and no
pre-trained model. The shape classifier listed below was a dropped experiment. It was trained on
pseudo-labels from our own v3 pipeline on this match, with no human labels.

`RESTART_OFFSET` is the one value measured on this match. It is the median first frame, counted from
the window start, with derived ball speed above 5 m/s in long windows. It uses no ground truth.
`measured_offset()` recomputes it on every run and prints it next to the value in use. Derived speed
is noisy on single frames, so a position glitch can trigger this measure in an individual window.

## What we tried and dropped

The scripts for these experiments are not in the repository, except the broadcast fill and the
pass-chain note in `possession_v2.py`. None of them feeds the submitted file.

| idea | result | why dropped |
|---|---|---|
| Fill ball gaps from broadcast | Coverage 59.3% → 63.2% | Exams did not improve; kept as an exam |
| Pass-chain score | Changed no frames | A chain with no opposing touch already stays with its team |
| Phases only where the ball is missing | Hand-over exam 0.7988 → 0.8039 | Partly inflated, because the pseudo-truth contains no phases; independent exams were neutral |
| Kinematic ensemble for gaps | 0.8246 on simulated gaps, about +0.01 | Gain was doubtful on real gaps |
| Shape classifier for possession | AUC 0.906 | No gain once the other changes were in |
| Frame-by-frame restart detector | Moved almost the same frames as the constant | Fired in 53% of normal play stretches |

## How we checked it

Ground truth exists for only five minutes, and the rules say that window is not for tuning. We
therefore compared versions with label-free exams.

- **`compare_versions.py`.** Cross-validation on alternating 60 s blocks. It measures agreement with
  held-out phases, and with control seen only in the broadcast ball.
- **`exam_handover.py`.** Places real gap patterns over well-covered minutes and asks each method to
  recover the possession changes. The answer is a pseudo-truth from direct control. It was used to
  compare ways of filling gaps.

| version | phases exam | ball exam | possessions | under 2 s | D frames |
|---|---:|---:|---:|---:|---:|
| v3 | 0.7480 | 0.7778 | 236 | 6.8% | 74,770 |
| v4 | 0.7621 | 0.7942 | 252 | 6.3% | 63,101 |

Exam scores are means over two folds. Possessions, short share and D frames come from the second
fold's labels. Exam frames include the public window, but these exams read no ground truth.

These exams are proxies. Agreement rises when a method makes fewer changes, so they must be read
next to the number of possessions, and neither one estimates `spellF1`. v4 improves both exams
while predicting more possessions, which works against that bias.

As a final pipeline check, we ran the official scorer on the public window after v4's code was
fixed. No parameter changed afterwards.

| entry | spellF1 | macroF1 |
|---|---:|---:|
| v4 | 0.447 | 0.822 |
| `baseline_smooth125` | 0.424 | 0.581 |
| v1 | 0.365 | 0.789 |
| v2 | 0.364 | 0.778 |
| v3 | 0.364 | 0.773 |
| `baseline_smooth50` | 0.326 | 0.621 |

**This is not an estimate of the final score.** Noise on this window is about ±0.33 for `spellF1`,
so the difference between v4 and `baseline_smooth125` means nothing at that scale.

## Limitations

- **Full-match `spellF1` is unknown.** Our only direct measure of it is the noisy public window.
- **The ball is missing on 41% of graded frames.** There, only phases and the evidence on either
  side of the gap speak. Changes that happen inside a gap are misplaced or missed.
- **The restart time is a single constant.** Its P10–P90 range across windows is about 2–11.5 s, so
  it is right on average and wrong for many individual restarts.
- **The constant fits this match.** We have no second match to test whether it generalises.
- **The taker score hides real changes.** It replaces all other evidence for about 11 s in a long
  window, so no change of possession can be placed inside that span.
- **Some taker information is lost.** When the restart falls inside a calm stretch, the live stretch
  starts over 1 s later and loses its known starting team. This affects 14 of 51 long windows. Any
  taker score left before mid-window still applies. Windows without a team get no taker score.
- **Control is a distance rule.** It cannot tell a player in control from one the ball is passing.
- **A stoppage resets the decoder.** A short live stretch with no named restart can flip team on a
  small amount of evidence, because a change across a stoppage is free.
- **The second-half kickoff is two frames early.** Our rule uses frame 113,740, while the FAQ
  publishes 113,742. We left this as it is, since it affects two frames of one possession and we
  froze the method after the final check.

With more time, we would detect the ball and the restart from the video, to reduce gaps and replace
the constant, and validate the method on other matches.

## LLM use

The code and this document were written with help from Claude Code. Match data, video and ground
truth were not shared with it. It saw the code, the formats published in this repository, and
summary numbers from our runs, such as scores and frame counts.

## Reproduction

Python 3.11 is required, because the pinned packages have no builds for much newer versions.

```bash
pip install -r solution/requirements.txt
python solution/possession_v4.py --out evaluation/results/submission.csv
python evaluation/validate_submission.py evaluation/results/submission.csv
```

The method is deterministic. Expected file: 199,319 rows; H 59,106, A 36,650, D 63,101, X 40,462;
247 possessions by the count in `validate_submission.py`.

SHA-256: `aeb23783192600b5caf24484a70c9f169bcb07424cb0734f71fe8996d6cc0308`

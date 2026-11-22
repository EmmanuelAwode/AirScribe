# AirScribe

Write and draw in the air with your index finger. Ink appears over a live, mirrored
webcam feed. Hand tracking runs locally with MediaPipe Hand Landmarker; nothing is
sent anywhere.

## Setup (Windows, PowerShell)

Requires 64-bit Python 3.12 (tested; 3.10–3.12 work with these pins) and a webcam.

```powershell
cd path\to\AirScribe
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\download_model.py
```

If `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first,
or call `.\.venv\Scripts\python.exe` directly instead of `python`.

### The model file

AirScribe needs `models\hand_landmarker.task` (about 7.5 MB, float16, version 1) from
Google's official MediaPipe model bucket. `scripts\download_model.py` fetches it. To do
it by hand instead:

```powershell
New-Item -ItemType Directory -Force models
Invoke-WebRequest https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -OutFile models\hand_landmarker.task
```

Or pass any other location with `--model path\to\hand_landmarker.task`.

## Run

```powershell
python -m airscribe
```

Options:

| Flag | Default | Purpose |
|---|---|---|
| `--camera N` | `0` | Webcam index. Try `1`, `2` if you have several cameras. |
| `--width`, `--height` | `1280`, `720` | Requested camera resolution (the camera may pick another). |
| `--model PATH` | `models\hand_landmarker.task` | Hand Landmarker model. |
| `--export-dir DIR` | `exports` | Where saved PNGs go. |
| `--log FILE.csv` | off | Write one row per frame (pinch values, pen state, why each stroke ended) for tuning. |
| `--no-mirror` | off | Start with the true (unmirrored) camera view. Toggle any time with **M** or the Mirror button. |

If the camera can't open, AirScribe prints a clear error and exits with code 1.
Quitting (Q, Esc, or the window's X) always releases the camera.

## Using it

- **Hover:** your index fingertip is the cursor. The small dot marks exactly where ink lands.
- **Draw:** pinch thumb and index together to put the pen down; open them to lift it.
- **No-pinch fallback:** press **Space** to turn on *key ink*. While it's on, the
  fingertip inks whenever a hand is visible. Press Space again to lift the pen.
- **Tools:** use the **mouse** on the toolbar. Your finger can't press toolbar buttons, so
  you can't change tools by accident while writing.
- **Status** (top right): `No hand detected`, `Ready`, `Drawing` or `Erasing`. The small
  `p 0.42` number next to it is the live pinch ratio, which helps when you tune thresholds.

### Keyboard controls

| Key | Action |
|---|---|
| Space | Toggle key ink (draw without pinching) |
| P / E | Pen / Eraser |
| 1–8 | Pick colour (white, black, red, orange, yellow, green, blue, purple) |
| `[` or `-` / `]` or `=` | Smaller / larger brush |
| U or Ctrl+Z | Undo last stroke (including an eraser stroke) |
| C | Clear all, then Y/Enter to confirm or N/Esc to cancel |
| S or Ctrl+S | Save ink as a transparent PNG |
| B | Switch between the mirrored camera and a plain canvas (ink is kept) |
| L | Show or hide hand landmarks |
| M | Mirror on/off: flip the camera like a mirror (default) or show the true camera view |
| Q / Esc | Quit |

## How a hand landmark becomes a stroke

```
camera frame ──flip──> HandTracker ──21 landmarks──> pinch ratio ──> PinchDetector ─┐
 (tracking.py)            (MediaPipe)           │      (gestures.py)    (hysteresis)  │ pen down?
                                                └─ index tip ──> One Euro smoother ───┤
                                                                   (smoothing.py)     ▼
                                       InkCanvas  <── begin/add/end ── StrokeController
                                       (canvas.py)                      (strokes.py)
                                           │
                       composite over camera or paper ──> toolbar + status ──> window
                                                            (ui.py, app.py)
```

1. **Mirror, then track.** Each frame is flipped horizontally *before* tracking (switch this off with M or the **Mirror** button if you prefer the true camera view), so the
   landmarks are already in screen space and moving your hand right moves the cursor right.
   MediaPipe returns 21 normalised landmarks; `tracking.py` turns them into pixels.
2. **Pinch, measured relative to hand size.** `pinch_ratio = |thumb_tip − index_tip| / |wrist − middle_knuckle|`.
   Dividing by hand size makes it work at any distance from the camera. It's measured two ways,
   and the app averages them:
   - **Tilt-corrected 2D** (`gestures.pinch_ratio_tilt_corrected`): the thumb–index gap in
     pixels, converted to metres with the palm segment the camera sees most face-on.
     Plain 2D blows up when the hand tilts towards the camera, because the on-screen hand
     shrinks; a pinch then reads 0.38 instead of 0.2.
   - **3D** from MediaPipe's world landmarks. It's steady to the side of the body. But
     when the hand turns in front of your body, the index finger hides the thumb tip, and a
     real pinch reads ~0.31.
   Averaging the two keeps the pinched/open split in the same place in every pose tested.
3. **Deciding pen down / up** (`gestures.PinchDetector`):
   - The ratio is averaged over the last 3 frames.
   - The pen goes down when it stays below the *start* level for 2 frames, and lifts when it
     stays above the *release* level for 2 frames. The gap between the two (hysteresis)
     stops flicker.
   - **It adapts to your hand.** People differ in where a relaxed thumb rests. If yours
     rests near your index finger, an open hand reads ~0.43, so a fixed release level of
     0.44 would never lift the pen between letters. The detector learns your open-hand
     value while the pen is up and keeps release 0.05 below it and start 0.17 below it.
     They never go higher than 0.44 / 0.28, and never lower than 0.30 / 0.22. It only
     learns from frames that are clearly open by the default levels. A real session
     showed why: when it learned from every pen-up frame, pinches that failed to register
     taught it a lower "open" value. That lowered the levels, so more pinches failed, and
     the open value sank from ~0.5 to 0.41.
   - A newly seen hand must be tracked for 3 frames before it can draw, so it leaves no
     stray dots.
   - While a release is pending, the pen holds still, so opening your fingers never adds a
     tail.
   - If the pen goes down late, the stroke reaches back up to 0.3 s through frames where
     your fingers were already clearly closed, so the start of the letter isn't lost.
   - After a tracking dropout of up to ~150 ms mid-stroke, a still-pinched hand draws again
     at once, as a **new** stroke.
4. **Smoothing.** The index tip goes through a One Euro filter. When the finger is slow,
   it smooths heavily to remove jitter. When the finger moves fast, the cutoff rises so
   corners and quick strokes stay sharp.
5. **Stroke control.** Every frame the `StrokeController` checks three things: is a hand
   visible, is the pen down, and is the fingertip inside the drawing area. If any answer is
   no, the current stroke **ends**. The next valid frame starts a **new** stroke. A
   single-frame jump larger than 25% of the frame diagonal is also treated as a glitch.
   So a tracking dropout can never draw a line across the canvas. When the hand
   disappears, the smoother is also reset, so the cursor doesn't glide in from the old position.
   The one exception is a **momentary pinch release**. If the pen goes back down within
   7 frames (~120 ms) and 40 px of where it lifted, the same stroke carries on, because a
   flicker like that isn't a real release. Moving to the next letter takes longer and goes
   further. A tracking dropout never re-joins.
6. **Ink layer.** Strokes are stored as point lists (that list is the source of truth) and
   rendered into a separate BGR colour image plus an alpha mask. The eraser is a stroke
   that writes alpha = 0. Undo pops the last stroke and re-renders, so you can undo an
   erase. The display blends the ink over the camera frame or a plain paper colour. Save
   writes only the colour and alpha layers to a BGRA PNG, so webcam pixels can never
   end up in the export.

## Tuning

All the numbers live in [`airscribe/config.py`](airscribe/config.py):

- `GestureConfig.pinch_start` / `pinch_release`: lower both if the pen goes down too
  easily; raise `pinch_release` if letters break; lower it if letters get joined. Watch the
  `p` value in the status bar while you pinch, or record a session with
  `python -m airscribe --log gestures.csv` and compare the `ratio2d` and `ratio3d` columns while pinched and open. The log also stores
  every hand landmark for each frame.
- `confirm_frames` (2), `warmup_frames` (3), `smooth_frames` (3): raise for steadier but
  slightly slower pen changes. `adapt`, `release_margin`, `start_margin`, `min_start`,
  `min_release`: how the thresholds follow your hand. `backfill_frames`, `resume_frames`:
  late-start recovery and dropout resume. `pinch_measure`: `"blend"` (default), `"2d"` or `"3d"`.
- Check any setting against a real recording without a camera:
  `python scripts\replay_log.py gestures.csv --start 0.28 --release 0.44`. It prints
  how many strokes the session would give, how many are tiny, and how much would ink.
  Add `--full replay.png` to re-run the whole app on the recorded hand landmarks and save
  the drawing it produces.
- `SmoothingConfig.min_cutoff`: lower it for steadier, laggier ink; raise it for snappier ink.
  `beta`: raise it if fast strokes lag or corners get rounded.
- `StrokeConfig.max_jump_fraction`, `edge_margin`, `min_point_distance`, and
  `rejoin_frames` / `rejoin_px` for re-joining a momentary release.
- `TrackerConfig.num_hands` (default 1) and the confidence thresholds.
- Palette, brush sizes, eraser scale.

## Tests

Everything below runs with no camera and no model file:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts\sim_report.py
```

- **Unit tests** (`tests/test_airscribe.py`) cover:
  - pinch thresholds, flicker resistance, confirmation, warm-up and adapting to the hand
  - stroke breaks: release, tracking loss, leaving the area, glitch jumps
  - smoothing, key-ink mode, eraser and undo, transparent PNG export, the view toggle, the toolbar
- **A simulated hand and webcam** (`tests/handsim.py`). It's a 21-landmark hand with adult
  proportions, posed, rotated, and projected through a 1280×720 pinhole camera. It
  reproduces three real error sources: 2D jitter, noisy 3D depth, and the thumb tip being
  mis-located when the index finger hides it. Its pinch values were calibrated against a
  real recorded session.
- **End-to-end writing tests** (`tests/writing_sim.py`, `tests/test_writing_sim.py`). The
  simulated hand writes **ROSH** through the real `AirScribe.process_hand` pipeline. It
  pinches, writes, opens its fingers and moves on, with its pose drifting. The resulting
  ink is scored for stroke count, share of the letters inked, and stray ink. Scenarios:
  - beside the body, in front of the body, tilted towards the camera, edge-on
  - fast writing, shaky and noisy, dim light with dropouts
  - one continuous trace around your own head
  - 20 seconds of hovering with the thumb half-closed, which must leave no ink
  A guard test checks that the old 3D-only settings still fail in front of the body, so
  the simulator keeps reproducing the real bug.
- **Real-session regressions**: `tests/data/session1_ratios.csv` and `session2_ratios.csv`
  hold pinch values from two real sessions (numbers only, no hand landmarks). They're
  replayed through the detector to check stroke count, accidental tiny strokes, and that
  the learned open level doesn't sink.
- `scripts\sim_report.py` prints the scenario table and saves `sim_report/summary.png`,
  which shows the worst run of each scenario before and after.

Results from `sim_report.py` (10 simulated writers each; "before" is the original 3D-only
pinch with fixed thresholds):

| Scenario | Strokes (ideal 6) before → now | Letters inked before → now |
|---|---|---|
| Beside body | 6.0 → 6.0 | 99.7% → 100% |
| In front of body | 12.0 → **6.0** | 72% → **98%** |
| Tilted towards camera | 6.1 → 6.0 | 99% → 100% |
| Shaky + noisy | 9.2 → **6.0** | 87% → **99%** |
| Dim light dropouts | 9.6 → 10.1 * | 89% → **96%** |
| Trace around yourself (ideal 1) | 2.4 → **1.0** | 94% → **100%** |

\* Strokes still split when tracking is lost; that's by design, so no line is ever drawn
across a gap.

Known limit: someone whose relaxed thumb rests very close to the index finger, writing in
front of their body with the thumb hidden, can still lose about 10% of a letter. To the
camera, their pinch and their relaxed hand look almost the same.

## Manual test checklist (needs a webcam)

1. `python -m airscribe` opens a window with a mirrored feed. Raise your right hand and the
   cursor moves to the right on screen. Status reads `Ready`; with no hand it reads `No hand detected`.
2. Pinch and write **R**, release, then write **O**, **S** and **H** the same way. You get four
   separate letters with no connecting lines between them.
3. Click a different colour swatch and draw. New ink uses the new colour; earlier ink is unchanged.
4. Pick **Eraser**, pinch over part of a letter, and confirm status shows `Erasing` and the ink disappears.
5. Click **Undo** once: the erased part comes back.
6. Click **Save**: a toast shows the path. Open the PNG in an editor that shows
   transparency and check it has only ink on a transparent background.
7. Move your hand out of frame mid-stroke, then bring it back while still pinching. A new
   stroke begins and nothing joins the two.
8. Hold a pinch near the threshold and watch the `p` value. Drawing shouldn't flicker on and off.
9. Press **Space**, hover-write without pinching, then press Space again to stop.
10. Press **B** to switch to the plain canvas; the drawing stays. **C**, then **N**, keeps the ink; **C**, then **Y**, clears it.
11. `python -m airscribe --camera 9` prints a clear camera error. Quit with Q and check
    the webcam light turns off.

## Project layout

```
airscribe/
  config.py     tunable thresholds, smoothing, palette
  tracking.py   MediaPipe Hand Landmarker wrapper (only file importing mediapipe)
  gestures.py   pinch ratio + hysteresis detector
  smoothing.py  One Euro filter
  strokes.py    stroke begin/extend/end rules
  canvas.py     ink layers, eraser, undo, PNG export
  ui.py         toolbar, status pill, cursor, confirm dialog
  app.py        main loop, input handling, rendering
scripts/download_model.py
scripts/replay_log.py     replay a --log recording with other pinch settings
scripts/sim_report.py     camera-free before/after report with pictures
tests/handsim.py          simulated hand + webcam
tests/writing_sim.py      simulated writing sessions scored end to end
tests/test_airscribe.py
```

Handwriting-to-text recognition is out of scope for this version.

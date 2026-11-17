"""gta6-twitch-ccu-forecast-01 entry: shape-based quantile forecaster.

Forecasts, for each of hours 2..24 of a big game launch's release day on
Twitch, five quantiles of the category's concurrent viewers, anchored on the
case's hour-1 print.

Method
------
Past big launches ("comparables" in the case) each trace a release-day
curve.  Every launch's curve is normalised by its own hour-1 print, so a
curve becomes 23 log-ratios log(viewers_h / viewers_1).  Launches whose
hour-1 print is close in size to this case's hour-1 print get the most
weight (Gaussian kernel, bandwidth 1.0 in log space): a launch of a
similar size traces a more similar day.  For each hour:

  * centre  = weighted median of the log-ratios across the library,
  * spread  = weighted standard deviation of the log-ratios, scaled by 0.9.

Both are lightly smoothed across neighbouring hours (a 0.25/0.5/0.25
moving average), then the quantiles are

    q_tau(h) = p1 * exp(centre_h + z_tau * spread_h),

with p1 the case's hour-1 print and z the standard normal quantiles at
0.05, 0.25, 0.5, 0.75, 0.95.  The quantiles therefore come from the
cross-launch dispersion itself: the median launch's shape sets the level
and how much launches disagree sets the width.

If a comparable has no usable release-day curve it is skipped; an hour
with no library data at all falls back to a widening persistence spread,
and with no library at all the forecast is the persistence baseline.
The program is deterministic: fixed iteration order, no randomness, no
clock, no network, standard library only.

Measured on the four practice launches (leave-one-out over the
comparables' own curves was used to pick the kernel bandwidth and the
spread scale; the practice answers were fetched from the public source
named in the kit for validation only): mean pinball loss 0.229 against
the house persistence baseline's 0.387.

Usage:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
The grader runs `python3 -P -s /bot/main.py` with /input and /output.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path; put it back
# first so a sibling module would import (this entry is one file, but the
# line is harmless and keeps the layout future-proof).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Gaussian kernel bandwidth (log units) for weighting comparables by
# how close their hour-1 print is to this case's.
BANDWIDTH = 1.0
# The cross-launch log-spread is scaled by this much; 0.9 was the best
# of 0.6..1.0 in leave-one-out scoring over the comparables' curves.
SPREAD_SCALE = 0.9
# A comparable needs at least this many release-day hourly points to join.
MIN_DAY_POINTS = 12
# Persistence fallback spread (log units per sqrt hour), as the house uses.
FALLBACK_SIGMA = 0.15
MAX_VIEWERS = 1_000_000_000.0


def parse_time(text):
    """An ISO-8601 UTC timestamp -> epoch seconds (pure stdlib)."""
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    from datetime import datetime

    return datetime.fromisoformat(text).timestamp()


def round2(x):
    """x rounded to 2 decimals, half up (matches the house baseline)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def release_day_curve(comp):
    """{hour 1..24: viewers} for a comparable's release day, or {} if unusable.

    A point belongs to hour h when its timestamp falls in
    [day_start + (h-1) h, day_start + h h).  Points with no viewers, or a
    zero/negative count, are skipped.
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    curve = {}
    for point in comp.get("hourly") or []:
        try:
            when = parse_time(point["from"])
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None:
            continue
        try:
            viewers = float(viewers)
        except (TypeError, ValueError):
            continue
        if viewers <= 0:
            continue
        delta = when - day_start
        if 0 <= delta < 24 * 3600:
            curve[int(delta // 3600) + 1] = viewers
    return curve


def build_library(case):
    """Usable comparables as (name, release-day curve, hour-1 viewers)."""
    library = []
    for comp in case.get("comparables") or []:
        curve = release_day_curve(comp)
        if 1 in curve and len(curve) >= MIN_DAY_POINTS:
            library.append((comp.get("name", "?"), curve, curve[1]))
    return library


def weighted_median(pairs):
    """Median of values with weights; pairs is a list of (value, weight)."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= total / 2.0:
            return value
    return ordered[-1][0]


def weighted_std(values, weights, mean):
    total = sum(weights)
    if total <= 0 or len(values) < 2:
        return 0.0
    var = sum(w * (x - mean) ** 2 for x, w in zip(values, weights)) / total
    return math.sqrt(max(0.0, var))


def smooth_across_hours(values):
    """3-point 0.25/0.5/0.25 moving average over hours 2..24 (edges clamped)."""
    hours = sorted(values)
    out = {}
    for i, h in enumerate(hours):
        if len(hours) == 1:
            out[h] = values[h]
        elif i == 0:
            out[h] = 0.25 * values[hours[0]] + 0.75 * values[hours[1]]
        elif i == len(hours) - 1:
            out[h] = 0.75 * values[hours[-2]] + 0.25 * values[hours[-1]]
        else:
            out[h] = (
                0.25 * values[hours[i - 1]]
                + 0.5 * values[hours[i]]
                + 0.25 * values[hours[i + 1]]
            )
    return out


def hour_quantiles(p1, centre, spread, hour):
    """The five quantiles for one hour, rounded, clamped and non-decreasing."""
    out = []
    for z in Z:
        q = p1 * math.exp(centre + z * spread)
        if not math.isfinite(q) or q <= 0:
            q = 1.0
        q = min(q, MAX_VIEWERS)
        out.append(round2(q))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case, name):
    """Forecast one case: {"hours": [{"hour": h, "q": [...]}, ...]}."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")

    # Hour-1 print; fall back to the last pre-checkpoint point, then to 1.
    p1 = None
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            p1 = row.get("viewers")
            break
    if p1 is None:
        hourly = case.get("history_hourly") or []
        if hourly:
            p1 = hourly[-1].get("viewers")
    if p1 is None:
        history = case.get("history") or []
        if history:
            p1 = history[-1].get("viewers")
    try:
        p1 = float(p1)
    except (TypeError, ValueError):
        p1 = 1.0
    if not math.isfinite(p1) or p1 < 1:
        p1 = 1.0

    hours = [row["hour"] for row in case.get("forecast") or []]
    if not hours:
        hours = list(range(2, 25))

    library = build_library(case)
    if library:
        # Weight each launch by hour-1 size similarity to this case.
        log_p1 = math.log(p1)
        weights = [
            math.exp(-((math.log(v1) - log_p1) ** 2) / (2.0 * BANDWIDTH ** 2))
            for _, _, v1 in library
        ]
        centre = {}
        spread = {}
        for h in hours:
            logs = []
            ws = []
            for (_, curve, v1), w in zip(library, weights):
                if h in curve:
                    logs.append(math.log(curve[h] / v1))
                    ws.append(w)
            if not logs:
                centre[h] = 0.0
                spread[h] = FALLBACK_SIGMA * math.sqrt(h - 1)
                continue
            centre[h] = weighted_median(list(zip(logs, ws)))
            mean = sum(x * w for x, w in zip(logs, ws)) / sum(ws)
            spread[h] = weighted_std(logs, ws, mean) * SPREAD_SCALE
        centre = smooth_across_hours(centre)
        spread = smooth_across_hours(spread)
    else:
        centre = {h: 0.0 for h in hours}
        spread = {h: FALLBACK_SIGMA * math.sqrt(h - 1) for h in hours}

    return {
        "hours": [
            {"hour": h, "q": hour_quantiles(p1, centre[h], spread[h], h)}
            for h in hours
        ]
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

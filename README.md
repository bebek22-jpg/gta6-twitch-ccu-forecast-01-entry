# GTA VI Twitch CCU forecast — shape-based quantile forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bebek22-jpg/gta6-twitch-ccu-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bebek22-jpg/gta6-twitch-ccu-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for the [gta6-twitch-ccu-forecast-01 bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01): forecast how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (2026-11-19 UTC), as five quantiles per hour, from the first hour's viewer print.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads one case
from the input directory and writes one forecast per case. It normalises
every past big launch in the case's `comparables` by its own hour-1 print,
then, for each hour 2–24, takes the weighted median of the log-ratios
(log viewers_h / viewers_1) across launches — weighted by how close each
launch's hour-1 size is to this case's — and sets the spread from the
weighted cross-launch dispersion, lightly smoothed across neighbouring
hours. It also corrects the centre for launch size: the log-ratio's slope
on log hour-1 viewers (near zero in the morning, about −0.9 by hour 20 —
small launches surge harder into the evening), shrunk by 0.75 and capped
at ±2 log units, applied only when the case's hour-1 print is below
50,000. Quantiles are the hour-1 print times exp(centre + z · spread).

## How to run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Validation

On the four practice launches (Wolverine, Red Dead Redemption 2, God of War
Ragnarök, Spider-Man 2), scored against their recorded release-day prints,
this entry's mean pinball loss on the log scale is **0.229**, against the
house persistence baseline's **0.387** (typical miss 48%, 90% coverage 0.95).
The kernel bandwidth, spread scale and size-slope shrinkage were picked
by leave-one-out scoring over the comparables' own release-day curves
(loss 0.390 vs baseline 0.930 there — the size-slope correction moved it
from 0.402), not by tuning on the practice cases.

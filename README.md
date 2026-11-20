# makefun-gta6-trophy-rarity-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/whatever7410258/makefun-gta6-trophy-rarity-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/whatever7410258/makefun-gta6-trophy-rarity-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI (PS5) trophy's earn rate on Exophase at 2026-11-26T00:00:00Z. A count-based dilution + progression model: per trophy it fits earners against players-tracked and time over the case's recent reads (new joiners' marginal earn rate vs. the existing base maturing), extrapolates players-tracked to the target, and reports five quantiles on the logit scale. When the history is flat (launch night, one unrefreshed page) it falls back to the refresh prior the rules state: one refresh, 638 → 3,868 players, new cohort earning a small fraction of the launch rate. The 100% trophy is forecast tight just under 100%.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=w9y8tk6d&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

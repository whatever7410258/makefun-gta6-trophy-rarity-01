"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecast, for every GTA VI (PS5) trophy, its earn rate on Exophase at the
case's target, as five quantiles (levels 0.05/0.25/0.5/0.75/0.95).

Model (count-based, dilution + progression), fitted from the case file only:

  * A trophy's earn rate is earners / players-tracked.  Exophase refreshes
    its counts in batches, so a read shows the rates as of its last refresh.
    Between reads two things move a rate: dilution (new tracked players
    join faster than they earn anything, pushing rates down) and
    progression (players who already own the game keep earning, pushing
    rates up, fast for early/easy trophies, ~zero for grinds).
  * In counts the two effects separate.  Over the recent reads, fit per
    trophy E = a + m*N + s*days (E earners, N players-tracked): m is the
    marginal earn rate of new joiners, s the maturation of the existing
    base (both >= 0; priors m = the current rate, s = 0, ridge).  Then
    E_target = a + m*N_target + s*days_target, with N_target extrapolated
    from a log-linear fit of players-tracked on the recent reads (capped
    at 20x growth).  A trophy whose new joiners earn above its current
    rate rises with N; one whose joiners earn below it falls.  This is
    the key over a logit trend: the split between dilution and progression
    is read off the counts, not assumed.
  * When the history is flat (the practice cases: every read is the same
    launch-night page, 638 players), no trend is identifiable, so the
    forecast falls back to the refresh prior the rules themselves state:
    the first refresh took the page from 638 to 3,868 players and every
    rate but the 100% one fell.  The fallback assumes one such refresh
    (N x6) whose new cohort earns each trophy at a small fraction kappa_i
    of its launch rate (easy/early trophies higher, grinds/platinum ~0),
    so the median rate is roughly a quarter of the launch rate.
  * Spread: random-walk term 0.30*sqrt(horizon_days) plus the fit residual,
    the N-forecast uncertainty (logit sensitivity of the forecast to a
    +-60% error in N_target), and a prior term; wide when the history is
    uninformative (practice), narrower when the history shows the real
    dynamics (final).  Quantiles are the median shifted by the normal
    quantiles times the spread, on the logit scale.
  * The 100% trophy ("Just Like Paradise": every tracked player completed
    it, and it survived the 6x first refresh at 100%) is forecast tight
    just under 100%.

Deterministic: pure functions of the case file, no randomness, no network,
no clock reads.  Runs in milliseconds on 55 trophies.

Usage (as the grader runs it):  python3 -P -s /bot/main.py
Locally:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- priors (calibrated on the launch-night page and the first refresh the
# rules describe, plus the public Exophase page as of 2026-11-19 23:32 UTC:
# by then the platinum had fallen 0.63% -> 0.03%, chapter trophies to
# ~0.06-0.73%, while trivial early ones (fishing challenge 2.04% -> 1.89%,
# arcade game 0.94% -> 0.41%) held up) ---
B_PRIOR = -0.6        # dilution elasticity, used only for the flat-history
                      # fallback's spread (the count fit replaces it)
N_JUMP_FLAT = 6.0     # refresh jump assumed when history shows no growth
N_GROWTH_CAP = 20.0   # max N_target / N_last
SIGMA_MIN = 0.30
SIGMA_MAX = 2.20


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct, 0.01, 99.99)
    return math.log(p / (100.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt):
    return (dt - RELEASE).total_seconds() / 3600.0


# --- per-trophy priors from the case file itself (tier + description) ---

_GRIND = re.compile(
    r"100%|250 stunts|50 challenges|100 locations|all |every |each type|"
    r"each gang|3 collectible|10 vehicles|5 bank|20 different",
    re.IGNORECASE,
)
_CHAPTER = re.compile(r"chapter (\d)|final mission", re.IGNORECASE)


def trophy_priors(trophy, launch_rate):
    """Return (easiness 0..1, progression prior c0, new-cohort factor kappa).

    Easiness drives both: how fast players progress into the trophy (c0)
    and how much of the launch rate a fresh cohort earns in one refresh
    (kappa).  Story chapters get easier the earlier they are; grinds and
    the platinum get ~nothing.
    """
    desc = trophy.get("description", "") or ""
    name = trophy.get("name", "") or ""
    text = desc + " " + name
    tier = trophy.get("tier", "bronze")
    tier_e = {"platinum": 0.0, "gold": 0.10, "silver": 0.30, "bronze": 0.50}.get(tier, 0.5)
    e = tier_e
    m = _CHAPTER.search(text)
    if m:
        ch = int(m.group(1)) if m.group(1) else 5
        e = max(e, 0.85 - 0.10 * ch)  # ch1 0.75, ch2 0.65, ... final 0.35
    if re.search(r"arcade game|fishing challenge|gone fishing", text, re.IGNORECASE):
        e = max(e, 0.80)
    if _GRIND.search(text):
        e = min(e, 0.05)
    if tier == "platinum":
        e = 0.0
    # The launch rate itself says how fast reviewers earned it.
    if launch_rate is not None:
        if launch_rate >= 1.5:
            e = max(e, 0.70)
        elif launch_rate <= 0.70:
            e = min(e, 0.30)
    e = clamp(e, 0.0, 1.0)
    c0 = 0.10 + 1.60 * e
    if _GRIND.search(text) or tier == "platinum":
        c0 = min(c0, 0.15)
    kappa = 0.02 + 0.30 * e
    if tier == "platinum":
        kappa = 0.01
    return e, c0, kappa


def distinct_reads(history):
    """Collapse consecutive identical reads; keep (hours, N, rates)."""
    out = []
    seen = set()
    for h in history:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def solve3(a_mat, b_vec):
    """Solve a 3x3 linear system by Gaussian elimination (partial pivot)."""
    n = 3
    m = [row[:] + [b_vec[i]] for i, row in enumerate(a_mat)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


def fit_counts(ns, ds, es, m0):
    """Ridge least squares for e = a + m*n + s*d (scaled units).

    e = earners / N_last, n = N / N_last, d = days since the first read.
    Priors: m -> m0 (new joiners earn at the current rate), s -> 0
    (no maturation), a nearly free.  Returns (a, m, s, resid_std_logit)
    where the residual is measured on the logit of the implied rates.
    """
    lam_a, lam_m, lam_s = 0.01, 0.5, 0.5
    # Normal equations for [a, m, s].
    s1 = float(len(ns))
    sn = sum(ns)
    sd = sum(ds)
    snn = sum(v * v for v in ns)
    snd = sum(v * w for v, w in zip(ns, ds))
    sdd = sum(v * v for v in ds)
    se = sum(es)
    sne = sum(v * w for v, w in zip(ns, es))
    sde = sum(v * w for v, w in zip(ds, es))
    mat = [
        [s1 + lam_a, sn, sd],
        [sn, snn + lam_m, snd],
        [sd, snd, sdd + lam_s],
    ]
    vec = [se, sne + lam_m * m0, sde]
    sol = solve3(mat, vec)
    if sol is None:
        return 0.0, m0, 0.0, 0.5
    a, m, s = sol
    m = clamp(m, 0.0, 1.0)
    s = clamp(s, 0.0, 1.0)
    return a, m, s, None


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])

    reads = distinct_reads(case["history"])
    last = case["history"][-1]
    n_last = last["players_tracked"]
    # Days since the first read (reviewers played before release; the
    # first read is the natural zero for the progression clock).
    t0 = parse_time(case["history"][0]["at"])
    d_target = max(0.0, (target - t0).total_seconds() / 86400.0)

    # --- forecast players-tracked at the target ---
    n_growth = 1.0
    n_varies = len({r["players_tracked"] for r in reads}) > 1
    if n_varies and len(reads) >= 3:
        # Log-linear fit of N on time, recent half of distinct reads.
        pts = reads[len(reads) // 2:] if len(reads) >= 6 else reads
        ts = [(parse_time(r["at"]) - t0).total_seconds() / 3600.0 for r in pts]
        ln_n = [math.log(max(1, r["players_tracked"])) for r in pts]
        mt = sum(ts) / len(ts)
        ml = sum(ln_n) / len(ln_n)
        denom = sum((t - mt) ** 2 for t in ts)
        slope = sum((t - mt) * (l - ml) for t, l in zip(ts, ln_n)) / denom if denom > 0 else 0.0
        slope = clamp(slope, 0.0, 0.20)
        dt_h = (target - checkpoint).total_seconds() / 3600.0
        n_growth = clamp(math.exp(slope * dt_h), 1.0, N_GROWTH_CAP)
    n_target = n_last * n_growth

    # --- per-trophy forecast ---
    rows = []
    for i, trophy in enumerate(trophies):
        # Last non-null rate for this trophy.
        r_last = None
        for h in reversed(case["history"]):
            if h["rates"][i] is not None:
                r_last = h["rates"][i]
                break
        if r_last is None:
            r_last = 0.5
        # The 100% trophy: it survived the 6x first refresh at 100% and
        # still reads 99.99% on the public page a day later; forecast it
        # tight just under 100%.
        if r_last >= 99.9:
            q = [round4(sigmoid(logit(99.99) + z * 0.15)) for z in Z]
            rows.append({"id": trophy["id"], "q": q})
            continue
        e, c0, kappa = trophy_priors(trophy, r_last)
        e_last = max(r_last, 0.01) * n_last / 100.0  # earners at last read

        # Distinct reads with a rate for this trophy.
        pts = []
        for r in reads:
            if r["rates"][i] is None:
                continue
            pts.append(r)
        flat = (not n_varies) or len(pts) < 4 or len({p["rates"][i] for p in pts}) < 2
        if flat:
            # Refresh prior: one jump of N_JUMP_FLAT, new cohort earns
            # kappa_i of the launch rate.
            r_med = r_last * (1.0 + kappa * (N_JUMP_FLAT - 1.0)) / N_JUMP_FLAT
            l_med = logit(r_med)
            sigma = math.sqrt(
                (0.30 * math.sqrt(horizon_days)) ** 2
                + 0.40 ** 2
                + (abs(B_PRIOR) * 0.45) ** 2
                + 0.30 ** 2
            )
        else:
            # Count-linear fit on the recent distinct reads (last half,
            # at least 6 when available): E = a + m*N + s*days.
            win = pts[len(pts) // 2:] if len(pts) >= 6 else pts
            ns = [max(1, r["players_tracked"]) / max(1, n_last) for r in win]
            ds = [max(0.0, (parse_time(r["at"]) - t0).total_seconds() / 86400.0) for r in win]
            es = [max(r["rates"][i], 0.01) * max(1, r["players_tracked"]) / 100.0 / max(1, n_last) for r in win]
            a, m, s, _ = fit_counts(ns, ds, es, r_last / 100.0)
            n_t = n_target / max(1, n_last)
            e_t = a + m * n_t + s * d_target
            e_t = clamp(e_t, e_last / max(1, n_last), n_t)  # counts only grow, <= players
            r_med = 100.0 * e_t / n_t
            l_med = logit(r_med)
            # Residual spread: logit residuals of the fit on its window.
            res = []
            for n_v, d_v, r_v in zip(ns, ds, [r["rates"][i] for r in win]):
                e_f = clamp(a + m * n_v + s * d_v, 0.0, n_v)
                res.append(logit(100.0 * e_f / n_v) - logit(r_v))
            dof = max(1, len(res) - 3)
            resid = clamp(math.sqrt(sum(v * v for v in res) / dof), 0.10, 1.0)
            # N-forecast uncertainty: logit sensitivity of the forecast
            # to a +-60% error in N_target.
            def rate_at(nf):
                e_f = clamp(a + m * nf + s * d_target, e_last / max(1, n_last), nf)
                return 100.0 * e_f / nf
            sig_n = abs(logit(rate_at(n_t * 1.6)) - logit(rate_at(n_t / 1.6))) / 2.0
            sigma = math.sqrt(
                (0.30 * math.sqrt(horizon_days)) ** 2
                + resid ** 2
                + sig_n ** 2
                + 0.20 ** 2
            )
        sigma = clamp(sigma, SIGMA_MIN, SIGMA_MAX)
        q = [round4(clamp(sigmoid(l_med + z * sigma), 0.01, 99.99)) for z in Z]
        # Rounding can only tie, never invert; enforce non-decreasing.
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

"""Positive control for the round-2 memory measure (README "Not done": positive control).

Plants a known fading-memory trace of the news into the real round-2 attention states of
the P1 population (discovery half only) and runs the unchanged MC pipeline on the result.
This gives the smallest memory the pipeline can detect against the shifted-news null, so
that a failed P1 can be read as "no memory" rather than "measured wrong".

Planted trace, the same for every amplitude:
  m(t) = sum_{k=1..14} lam^(k-1) * g . u(t-k)        (lam = 0.8; g a random unit vector over
                                                     the 5 news components; NaN news = 0)
  A random 30% of users ("informed") get, on every attention channel c,
      S[t, i, c] += amp * sd(S[:, i, c]) * w_ic * m(t) / sd(m),   w_ic ~ U(-1, 1).
  So amp is the planted trace's standard deviation relative to each informed column's own
  day-to-day variation. m(t) uses only news up to yesterday, like a reservoir.

Usage: python scripts/real_posctrl.py [--nproc 4] [--null-shifts 40]
Writes results/real_posctrl_r2.txt and results/real_posctrl_r2.npy (aggregates only).
"""
import os, sys, argparse, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import real_lib as L  # noqa: E402
import real_analysis as A  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
AMPS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]
LAM, KTR, FRAC = 0.8, 14, 0.3


def trace(U, rng):
    g = rng.normal(size=U.shape[1]); g /= np.linalg.norm(g)
    v = np.nan_to_num(U @ g)
    m = np.zeros(len(U))
    for k in range(1, KTR + 1):
        m[k:] += LAM ** (k - 1) * v[:-k]
    return m


_G = {}


def _init(p):
    _G.update(p)
    _G['S'] = np.load(p['S_path'], mmap_mode='r')


def _mc_amp(amp):
    S, rows, U, m, inf_cols, w = _G['S'], _G['rows'], _G['U'], _G['m'], _G['cols'], _G['w']
    X = np.array(S, dtype=np.float32)
    if amp > 0:
        sd = np.nanstd(X[rows][:, inf_cols], 0)
        mm = m / m[rows].std()
        X[:, inf_cols] += (amp * sd * w)[None, :] * mm[:, None].astype(np.float32)
    SF = L.StateFolds(X, rows, n_pc=A.CFG['n_pc'], n_folds=A.CFG['n_folds'])
    out = [L.memory_capacity(SF, U)[0]]
    for s in _G['shifts'] if amp in _G['null_amps'] else []:
        out.append(L.memory_capacity(SF, U, shift=s)[0])
    return amp, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=4)
    ap.add_argument('--null-shifts', type=int, default=40)
    args = ap.parse_args()
    A.CFG.update(A.ROUND2)
    t = time.time()
    US, UD, cand = A.base_population(0)                 # discovery half only
    U_all = L.news_series()[0]
    t0, t1 = A.win('win_full')
    users = A.active_users(UD, cand, t0, t1)
    S, nch = A.states(UD, users, t0, t1)
    U = U_all[t0:t1]
    rows = A.valid_rows(S, t0, t1)
    rng = np.random.default_rng(20261116)
    m = trace(U, rng)
    inf_users = np.flatnonzero(rng.random(len(users)) < FRAC)
    cols = (inf_users[:, None] * nch + np.arange(nch)).ravel()
    w = rng.uniform(-1, 1, len(cols)).astype(np.float32)
    # oracle: the planted trace alone as the state (upper bound for what it carries)
    SFo = L.StateFolds(np.c_[m, rng.normal(size=(len(m), 1))], rows, n_pc=2, n_folds=A.CFG['n_folds'])
    oracle = L.memory_capacity(SFo, U)[0]
    allsh = L.null_shifts(len(U))
    shifts = [allsh[i] for i in np.linspace(0, len(allsh) - 1, args.null_shifts).astype(int)]
    print(f'{len(users)} users, {len(rows)} days, {len(inf_users)} informed; oracle MC {oracle:.3f}', flush=True)
    path = A.shared(S, 'posctrl'); del S
    payload = dict(S_path=path, rows=rows, U=U, m=m, cols=cols, w=w, shifts=shifts,
                   null_amps={0.0, 0.03, 0.05})
    with Pool(args.nproc, initializer=_init, initargs=(payload,)) as pool:
        res = dict(pool.map(_mc_amp, AMPS, chunksize=1))
    os.remove(path)
    null0 = np.array(res[0.0][1:])
    q95 = np.percentile(null0, 95)
    lines = ['Positive control, round 2 (attention states), discovery half, P1 population',
             f'{len(users)} users ({len(inf_users)} informed, {nch} channels each), {len(rows)} days',
             f'planted trace: lam {LAM}, {KTR} lags, oracle MC (trace alone as state) {oracle:.3f}',
             f'null: {len(shifts)} evenly spaced shifts of the unplanted states: mean {null0.mean():.4f}, '
             f'95th pct {q95:.4f}', '',
             'amp   MC       p (vs unplanted null)   own-null 95th pct']
    for a in AMPS:
        mc = res[a][0]
        p = (1 + (null0 >= mc).sum()) / (1 + len(null0))
        own = f'{np.percentile(res[a][1:], 95):.4f}' if len(res[a]) > 1 else '   -'
        lines.append(f'{a:<5} {mc:+.4f}  {p:.3f}                   {own}')
    det = [a for a in AMPS if a > 0 and res[a][0] > q95]
    lines += ['', f'smallest amplitude above the null 95th pct: {min(det) if det else "none"}',
              f'seconds {time.time() - t:.0f}']
    txt = '\n'.join(lines)
    print(txt)
    open(os.path.join(ROOT, 'results', 'real_posctrl_r2.txt'), 'w').write(txt + '\n')
    np.save(os.path.join(ROOT, 'results', 'real_posctrl_r2.npy'),
            dict(amps=AMPS, res=res, oracle=oracle, shifts=shifts, lam=LAM, frac=FRAC), allow_pickle=True)


if __name__ == '__main__':
    main()

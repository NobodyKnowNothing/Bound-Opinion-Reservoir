"""GDELT 1.0 daily exports -> daily news feature matrix (one row per day, 2023-02-17 .. 2024-03-18).

Usage: python -I scripts/real_gdelt.py data/gdelt data/derived/gdelt_features.npz

Per day (file = date events were *reported*): log event count, share of events in each of
the 20 CAMEO root codes, mean tone and mean Goldstein score, and the share of events located
in each of the 25 most frequent action countries over the window. Days 2023-03-22 (partial
file) and 2023-03-23 (missing) are NaN rows. The news series u(t) is made from these in
real_analysis.py.
"""
import sys, os, zipfile, io
from collections import Counter
from multiprocessing import Pool
import numpy as np

DAY0 = np.datetime64('2023-02-17')
NDAYS = 396
BAD_DAYS = {'20230322', '20230323'}
N_COUNTRIES = 25


def day_str(t):
    return str(DAY0 + np.timedelta64(t, 'D')).replace('-', '')


def read_day(path):
    roots = np.zeros(21); tone = gold = 0.0; n = 0
    countries = Counter()
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            for line in io.TextIOWrapper(f, encoding='utf-8', errors='replace'):
                c = line.split('\t')
                if len(c) < 58:
                    continue
                try:
                    r = int(c[28]); g = float(c[30]); a = float(c[34])
                except ValueError:
                    continue
                if 1 <= r <= 20:
                    roots[r] += 1
                n += 1; tone += a; gold += g
                countries[c[51]] += 1
    return n, roots[1:], tone / max(n, 1), gold / max(n, 1), countries


def main():
    src, out = sys.argv[1], sys.argv[2]
    days = [day_str(t) for t in range(NDAYS)]
    paths = [os.path.join(src, f'{d}.export.CSV.zip') for d in days]
    todo = [p for d, p in zip(days, paths) if d not in BAD_DAYS and os.path.exists(p)]
    with Pool(4) as pool:
        res = dict(zip(todo, pool.map(read_day, todo)))
    tot = Counter()
    for r in res.values():
        tot.update(r[4])
    top = [c for c, _ in tot.most_common(N_COUNTRIES + 1) if c][:N_COUNTRIES]
    names = (['log_events'] + [f'root{r:02d}' for r in range(1, 21)] + ['tone', 'goldstein']
             + [f'geo_{c}' for c in top])
    X = np.full((NDAYS, len(names)), np.nan)
    for t, p in enumerate(paths):
        if p not in res:
            continue
        n, roots, tone, gold, cs = res[p]
        X[t] = np.concatenate([[np.log(n)], roots / n, [tone, gold], [cs[c] / n for c in top]])
    np.savez(out, X=X, names=np.array(names), days=np.array(days))
    print('days', np.isfinite(X[:, 0]).sum(), 'features', len(names), 'countries', top)


if __name__ == '__main__':
    main()

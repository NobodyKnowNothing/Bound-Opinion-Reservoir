"""Shared pieces of the Bluesky / GDELT test (REAL_DATA.md, REAL_PREREG.md).

Inputs come from scripts/real_extract_posts.py, real_extract_edges.py and real_gdelt.py,
all under data/derived (git-ignored). Days are indexed from 2023-02-17 (day 0) to
2024-03-18 (day 395).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from real_extract_posts import U_DTYPE, UD_DTYPE  # noqa: E402

DERIVED = os.path.join(os.path.dirname(__file__), '..', 'data', 'derived')
NDAYS = 396
DAY0 = np.datetime64('2023-02-17')
BREAK = int((np.datetime64('2024-02-06') - DAY0).astype(int))      # public opening = day 354
LATE0 = int((np.datetime64('2023-10-01') - DAY0).astype(int))      # day 226
HP_WIN = 28          # causal high-pass: subtract mean of the previous 28 days
HP_MIN = 7           # need at least 7 previous days
BREAK_SKIP = 7       # rows dropped right after the break
K_MAX = 14
N_NEWS = 5
SPLIT_SEED = 20261111


def day_of(s):
    return int((np.datetime64(s) - DAY0).astype(int))


# ----------------------------------------------------------------------------- loading
USERS_FILE, USERDAYS_FILE = 'users.bin', 'userdays.bin'


def load_users():
    """Per-user summaries sorted by user id (complete records only)."""
    path = os.path.join(DERIVED, USERS_FILE)
    n = os.path.getsize(path) // U_DTYPE.itemsize
    U = np.fromfile(path, dtype=U_DTYPE, count=n)
    return U[np.argsort(U['user'], kind='stable')]


def load_userdays():
    path = os.path.join(DERIVED, USERDAYS_FILE)
    n = os.path.getsize(path) // UD_DTYPE.itemsize
    return np.memmap(path, dtype=UD_DTYPE, mode='r', shape=(n,))


def half_of(uid):
    """0 = discovery, 1 = confirmation. Fixed random split of user ids."""
    rng = np.random.default_rng(SPLIT_SEED)
    lab = (rng.random(int(uid.max()) + 1) < 0.5).astype(np.int8)
    return lab[uid]


# ----------------------------------------------------------------------------- filters
def causal_highpass(X, restart=BREAK):
    """x_t minus the mean of x over the previous HP_WIN days (NaN-aware), never reaching
    back across the break. Rows with fewer than HP_MIN finite previous days become NaN."""
    T = len(X)
    out = np.full(X.shape, np.nan, dtype=np.float64)
    fin = np.isfinite(X)
    Xz = np.where(fin, X, 0.0)
    cs = np.vstack([np.zeros((1,) + X.shape[1:]), np.cumsum(Xz, 0)])
    cn = np.vstack([np.zeros((1,) + X.shape[1:]), np.cumsum(fin, 0)])
    for t in range(1, T):
        lo = max(0, t - HP_WIN)
        if restart is not None and t >= restart:
            lo = max(lo, restart)
        n = cn[t] - cn[lo]
        m = (cs[t] - cs[lo]) / np.maximum(n, 1)
        out[t] = np.where(n >= HP_MIN, X[t] - m, np.nan)
    return out


def remove_weekday(X):
    dow = np.arange(len(X)) % 7
    X = X.copy()
    for d in range(7):
        X[dow == d] -= np.nanmean(X[dow == d], 0)
    return X


HOLIDAYS = ['2023-02-20', '2023-04-07', '2023-04-09', '2023-05-29', '2023-06-19', '2023-07-04',
            '2023-09-04', '2023-11-23', '2023-11-24'] +            [str(np.datetime64('2023-12-23') + np.timedelta64(i, 'D')) for i in range(11)] +            ['2024-01-15', '2024-02-19']


def holiday_mask(T=NDAYS):
    m = np.zeros(T, bool)
    for h in HOLIDAYS:
        t = day_of(h)
        if 0 <= t < T:
            m[t] = True
    return m


def news_series(n_comp=N_NEWS, mask_holidays=True):
    """u(t): leading PCs of the high-passed, weekday-adjusted GDELT features (unit variance).
    Holidays (US federal, Easter, 23 Dec - 2 Jan) are set to NaN afterwards: they move news
    volume and sentiment together for calendar reasons, so no row may use them."""
    z = np.load(os.path.join(DERIVED, 'gdelt_features.npz'))
    Xr = z['X'].copy()
    if mask_holidays:
        Xr[holiday_mask(len(Xr))] = np.nan          # keep holidays out of the trailing means too
    H = remove_weekday(causal_highpass(Xr, restart=None))
    ok = np.isfinite(H).all(1)
    Z = (H[ok] - H[ok].mean(0)) / H[ok].std(0)
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    U = np.full((len(H), n_comp), np.nan)
    U[ok] = Z @ Vt[:n_comp].T
    U /= np.nanstd(U, 0)
    if mask_holidays:
        U[holiday_mask(len(U))] = np.nan
    return U, (S ** 2 / (S ** 2).sum())[:n_comp], Vt[:n_comp], z['names']


# ----------------------------------------------------------------------------- states
def daily_opinion(UD, users, field='free', t0=0, t1=NDAYS):
    """(T, N) daily mean sentiment of each user's own English posts; NaN on silent days.
    field 'free' leaves out posts that mention the news lexicon; 'all' keeps them;
    'topic<k>' uses only posts on fixed topic k."""
    users = np.asarray(users)
    sel = np.isin(UD['user'], users) & (UD['day'] >= t0) & (UD['day'] < t1)
    R = UD[sel]
    col = np.searchsorted(users, R['user'])
    if field == 'free':
        n, s = R['n_free'], R['sum_free']
    elif field == 'all':
        n, s = R['n_sent'], R['sum_s']
    else:
        k = int(field[5:])
        n, s = R['tn'][:, k], R['ts'][:, k]
    X = np.full((t1 - t0, len(users)), np.nan, np.float32)
    N = np.zeros((t1 - t0, len(users)), np.int32)
    ok = n > 0
    X[R['day'][ok] - t0, col[ok]] = s[ok] / n[ok]
    N[R['day'] - t0, col] = n
    return X, N


def carry_forward(X):
    """Silent days keep the last expressed opinion; days before the first post get the
    user's mean."""
    X = X.copy()
    mean = np.nanmean(X, 0)
    last = np.where(np.isfinite(X[0]), X[0], mean)
    X[0] = last
    for t in range(1, len(X)):
        f = np.isfinite(X[t])
        last = np.where(f, X[t], last)
        X[t] = last
    return X


def user_states(X, t0=0):
    """carry-forward -> causal high-pass (restarting at the break) -> weekday adjustment.
    X starts at absolute day t0. Holiday posts are ignored (silent days), so they never enter
    the trailing means; holiday rows themselves are NaN."""
    hol = holiday_mask()[t0:t0 + len(X)]
    X = X.copy()
    X[hol] = np.nan
    S = causal_highpass(carry_forward(X).astype(np.float64), restart=BREAK - t0)
    S[hol] = np.nan
    dow = (np.arange(len(S)) + t0) % 7
    for d in range(7):
        S[dow == d] -= np.nanmean(S[dow == d], 0)
    return S


# ----------------------------------------------------------------------------- ridge
ALPHAS = np.logspace(-1, 5, 13)


def _std(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _ridge_preds(Xtr, Ytr, Xte, alphas):
    """Predictions on Xte for each alpha: (len(alphas), n_te, q). Inputs pre-standardized;
    intercept from the training mean of Y."""
    ym = Ytr.mean(0)
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    UtY = U.T @ (Ytr - ym)
    XV = Xte @ Vt.T
    out = []
    for a in alphas:
        out.append(XV @ ((s / (s ** 2 + a))[:, None] * UtY) + ym)
    return np.array(out)


def _blocks(idx, n_blocks):
    return np.array_split(idx, n_blocks)


def ridge_alpha(X, Y, train, inner=3, gap=K_MAX):
    """Alpha picked by inner blocked CV on the training rows."""
    sse = np.zeros(len(ALPHAS))
    for blk in _blocks(train, inner):
        lo, hi = blk.min() - gap, blk.max() + gap
        tr = train[(train < lo) | (train > hi)]
        if len(tr) < 20:
            continue
        a, b = _std(X[tr], X[blk])
        P = _ridge_preds(a, Y[tr], b, ALPHAS)
        sse += ((P - Y[blk]) ** 2).sum((1, 2))
    return ALPHAS[np.argmin(sse)]


def ridge_cv_predict(X, Y, train, test, inner=3, gap=K_MAX, also=None):
    """Fit on rows `train` with alpha picked by inner blocked CV, predict rows `test`
    (and rows `also`, if given)."""
    best = ridge_alpha(X, Y, train, inner, gap)
    a, b = _std(X[train], X[test])
    if also is None:
        return _ridge_preds(a, Y[train], b, [best])[0]
    c = _std(X[train], X[also])[1]
    return _ridge_preds(a, Y[train], b, [best])[0], _ridge_preds(a, Y[train], c, [best])[0]


# ----------------------------------------------------------------------------- memory capacity
class StateFolds:
    """Blocked time folds over the valid rows of a state matrix; the PCs of the user states
    are fitted on each fold's training rows only (users standardized on those rows)."""

    def __init__(self, S, rows, n_pc=20, n_folds=5, gap=K_MAX):
        self.rows = rows = np.asarray(rows)
        self.folds = []
        for test in _blocks(rows, n_folds):
            lo, hi = test.min() - gap, test.max() + gap
            train = rows[(rows < lo) | (rows > hi)]
            A = S[train]
            mu, sd = A.mean(0), A.std(0)
            sd[sd == 0] = 1
            A = (A - mu) / sd
            _, _, Vt = np.linalg.svd(A, full_matrices=False)
            m = min(n_pc, Vt.shape[0])
            Z = np.full((S.shape[0], m), np.nan)
            Z[rows] = ((S[rows] - mu) / sd) @ Vt[:m].T
            self.folds.append((train, test, Z))


def _stagewise(Z, B, Y, tr, te, cache=None, key=None):
    """Baseline ridge on the news since t-k; then a ridge from the state PCs to the baseline's
    training residuals. Returns (full prediction, baseline prediction) on rows te."""
    if cache is not None and key in cache:
        pb_te, pb_tr = cache[key]
    else:
        pb_te, pb_tr = ridge_cv_predict(B, Y, tr, te, also=tr)
        if cache is not None:
            cache[key] = (pb_te, pb_tr)
    R = Y.copy()
    R[tr] = Y[tr] - pb_tr
    return pb_te + ridge_cv_predict(Z, R, tr, te), pb_te


def memory_capacity(SF, U, k_max=K_MAX, shift=0, baseline_cache=None, method='stagewise'):
    """MC = sum_k [R2(u_{t-k} <- PCs(state_t), u_{t-k+1..t}) - R2(u_{t-k} <- u_{t-k+1..t})].
    R2 pooled over the news components, from out-of-fold predictions (blocked CV).
    `shift` circularly shifts the news series (null). Returns (MC, per-k increments,
    per-k full R2, per-k baseline R2)."""
    T = U.shape[0]
    Us = np.roll(U, shift, 0) if shift else U
    inc, full_r2, base_r2 = [], [], []
    for k in range(1, k_max + 1):
        lagged = [np.roll(Us, j, 0) for j in range(0, k)]          # u_t ... u_{t-k+1}
        B = np.concatenate(lagged, 1)
        Y = np.roll(Us, k, 0)                                       # u_{t-k}
        t = np.arange(T)
        okY = (t >= k) & np.isfinite(Y).all(1) & np.isfinite(B).all(1)
        pf, pb, yy = [], [], []
        for train, test, Z in SF.folds:
            tr, te = train[okY[train]], test[okY[test]]
            if len(te) == 0:
                continue
            if method == 'stagewise':
                f_, b_ = _stagewise(Z, B, Y, tr, te, baseline_cache,
                                    (shift, k, te[0], len(te), len(tr)))
                pf.append(f_); pb.append(b_); yy.append(Y[te])
                continue
            Xf = np.concatenate([Z, B], 1)
            pf.append(ridge_cv_predict(Xf, Y, tr, te))
            key = (shift, k, te[0], len(te), len(tr))
            if baseline_cache is not None and key in baseline_cache:
                pb.append(baseline_cache[key])
            else:
                p = ridge_cv_predict(B, Y, tr, te)
                if baseline_cache is not None:
                    baseline_cache[key] = p
                pb.append(p)
            yy.append(Y[te])
        yy, pf, pb = np.concatenate(yy), np.concatenate(pf), np.concatenate(pb)
        sst = ((yy - yy.mean(0)) ** 2).sum()
        rf, rb = 1 - ((yy - pf) ** 2).sum() / sst, 1 - ((yy - pb) ** 2).sum() / sst
        full_r2.append(rf); base_r2.append(rb); inc.append(rf - rb)
    return float(np.sum(inc)), np.array(inc), np.array(full_r2), np.array(base_r2)


def null_shifts(T, min_shift=28, step=1):
    """Every circular shift that moves the news at least 4 weeks. Weekday effects are already
    removed from both series, so shifts need not keep weekday alignment; step 1 gives ~T-56
    shifts, enough for a one-sided p below the Holm threshold of six tests (0.05/6)."""
    return [s for s in range(min_shift, T - min_shift + 1, step)]


# ----------------------------------------------------------------------------- stats
def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1, run)
    return adj

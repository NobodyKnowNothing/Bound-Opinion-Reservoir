"""Bluesky / GDELT test of the simulation's predictions (REAL_DATA.md; rules frozen in
REAL_PREREG.md).

Usage:
  python scripts/real_analysis.py describe  --half discovery      # counts only
  python scripts/real_analysis.py run       --half discovery      # all predictions
  python scripts/real_analysis.py run       --half confirm        # after pre-registration

Outputs results/real_<half>.txt, results/real_<half>.npz and figures/real_<half>.png.
Only aggregates are written; no post text is read here.
"""
import os, sys, argparse, json, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import real_lib as L  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')

# ----------------------------------------------------------------------------- frozen rules
CFG = dict(
    # P1/P2 window and the "active throughout" rule
    win_full=('2023-07-01', '2024-03-19'),
    win_late=('2023-10-01', '2024-03-19'),
    act_frac=0.6,           # >= 1 lexicon-free English post in at least 60% of the window's weeks
    eng_frac=0.5,           # >= 50% of own posts are English with a sentiment score
    watcher_min_news=3,     # >= 3 own posts carrying a news-site link -> watcher
    n_pc=20,                # user-state PCs fed to the ridge decoder
    n_folds=5,
    opinion_min_posts=10,   # posts in the late window needed to give an account an opinion
    p3_min_followees=10,    # followees with an opinion needed to enter prediction 3
    p3_groups=5,
    p4_strata=(1, 2, 3, 5, 9, 17),     # lower edges of interaction-day strata
    p4_news_frac=1 / 3,     # top / bottom third of late-window days by ||u_t||
    n_perm=200,
    n_boot=1000,
    field='free',
)
SEED = 7


def win(name):
    a, b = CFG[name]
    return L.day_of(a), L.day_of(b)


# ----------------------------------------------------------------------------- selection
def active_users(UD, cand, t0, t1):
    """Users in `cand` with a lexicon-free English post in >= act_frac of the weeks in [t0, t1)."""
    sel = np.isin(UD['user'], cand) & (UD['day'] >= t0) & (UD['day'] < t1) & (UD['n_free'] > 0)
    R = UD[sel]
    nweeks = int(np.ceil((t1 - t0) / 7))
    key = np.unique(R['user'].astype(np.int64) * 1000 + (R['day'] - t0) // 7)
    u, c = np.unique(key // 1000, return_counts=True)
    keep = u[c >= CFG['act_frac'] * nweeks].astype(np.int64)
    return np.sort(keep)


def activity(UD, users, t0, t1):
    """Per user: lexicon-free posts and days with such posts in [t0, t1)."""
    sel = np.isin(UD['user'], users) & (UD['day'] >= t0) & (UD['day'] < t1)
    R = UD[sel]
    col = np.searchsorted(users, R['user'])
    n = np.bincount(col, weights=R['n_free'], minlength=len(users))
    d = np.bincount(col, weights=R['n_free'] > 0, minlength=len(users))
    return n, d


def base_population(half):
    US = L.load_users()
    UD = L.load_userdays()
    eng = (US['n_sent'] >= 20) & (US['n_sent'] >= CFG['eng_frac'] * np.maximum(US['n_own'], 1))
    h = L.half_of(US['user'].astype(np.int64))
    cand = np.sort(US['user'][eng & (h == half)].astype(np.int64))
    return US, UD, cand


def roles(US, users):
    idx = np.searchsorted(US['user'], users)
    news = US['n_news'][idx]
    return news >= CFG['watcher_min_news'], news == 0


# ----------------------------------------------------------------------------- MC workers
_G = {}


def _init(payload):
    _G.update(payload)


def _mc_shift(shift):
    return L.memory_capacity(_G['SF'], _G['U'], shift=shift)[0]


def mc_with_null(S, U, rows, pool):
    SF = L.StateFolds(S, rows, n_pc=CFG['n_pc'], n_folds=CFG['n_folds'])
    mc, inc, rf, rb = L.memory_capacity(SF, U)
    shifts = L.null_shifts(len(U))
    pool_ = Pool(pool, initializer=_init, initargs=({'SF': SF, 'U': U},))
    null = np.array(pool_.map(_mc_shift, shifts)); pool_.close()
    p = (1 + (null >= mc).sum()) / (1 + len(null))
    return dict(mc=mc, inc=inc, full=rf, base=rb, null=null, p=p)


def _group_mc(cols):
    S, U, rows = _G['S'], _G['U'], _G['rows']
    SF = L.StateFolds(S[:, cols], rows, n_pc=CFG['n_pc'], n_folds=CFG['n_folds'])
    return L.memory_capacity(SF, U, baseline_cache=_G.setdefault('cache', {}))[0]


def _p2_perm(seed):
    rng = np.random.default_rng(seed)
    a, b = _G['pairs']
    sw = rng.random(len(a)) < 0.5
    ga, gb = np.where(sw, b, a), np.where(sw, a, b)
    return _group_mc(ga) - _group_mc(gb)


def _slope(groups_mc):
    x = np.arange(len(groups_mc), dtype=float)
    return np.polyfit(x, groups_mc, 1)[0]


def _p3_perm(seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(_G['n'])
    return _slope([_group_mc(g) for g in np.array_split(order, CFG['p3_groups'])])


def valid_rows(S, t0, t1):
    ok = np.isfinite(S).all(1)
    rows = np.flatnonzero(ok)
    return rows


# ----------------------------------------------------------------------------- P1, P2
def run_p1_p2(half, US, UD, cand, U_all, out, nproc):
    t0, t1 = win('win_full')
    users = active_users(UD, cand, t0, t1)
    watch, non = roles(US, users)
    X, _ = L.daily_opinion(UD, users, CFG['field'], t0, t1)
    S = L.user_states(X, t0)
    U = U_all[t0:t1]
    rows = valid_rows(S, t0, t1)
    out['p1_users'] = len(users)
    out['p1_rows'] = len(rows)
    print(f'P1: {len(users)} users, {len(rows)} days, {watch.sum()} watchers, {non.sum()} non-watchers', flush=True)
    r1 = mc_with_null(S, U, rows, nproc)
    out['p1'] = r1
    print(f"P1 MC {r1['mc']:.3f}  null mean {r1['null'].mean():.3f} 95% {np.percentile(r1['null'], 95):.3f}  p {r1['p']:.4f}", flush=True)

    # P2: watchers vs activity-matched non-watchers
    n_post, n_day = activity(UD, users, t0, t1)
    f = np.stack([np.log1p(n_post), n_day], 1)
    f = (f - f.mean(0)) / f.std(0)
    rng = np.random.default_rng(SEED)
    W = np.flatnonzero(watch); C = list(np.flatnonzero(non))
    pairs_a, pairs_b = [], []
    for i in rng.permutation(W):
        if not C:
            break
        d = ((f[C] - f[i]) ** 2).sum(1)
        j = int(np.argmin(d))
        pairs_a.append(i); pairs_b.append(C.pop(j))
    pa, pb = np.array(pairs_a), np.array(pairs_b)
    pool = Pool(nproc, initializer=_init, initargs=({'S': S, 'U': U, 'rows': rows, 'pairs': (pa, pb)},))
    mw, mn = pool.map(_group_mc, [pa, pb])
    diff = mw - mn
    perms = np.array(pool.map(_p2_perm, range(1000, 1000 + CFG['n_perm']))); pool.close()
    p = (1 + (perms >= diff).sum()) / (1 + len(perms))
    out['p2'] = dict(n_pairs=len(pa), mc_watch=mw, mc_non=mn, diff=diff, perms=perms, p=p,
                     act_watch=(n_post[pa].mean(), n_day[pa].mean()),
                     act_non=(n_post[pb].mean(), n_day[pb].mean()))
    print(f'P2 pairs {len(pa)}  MC watchers {mw:.3f} non {mn:.3f} diff {diff:.3f}  p {p:.4f}', flush=True)
    return users, watch, non


# ----------------------------------------------------------------------------- P3, P4
def late_opinions(UD, t0, t1):
    sel = (UD['day'] >= t0) & (UD['day'] < t1)
    R = UD[sel]
    m = int(R['user'].max()) + 1
    n = np.bincount(R['user'], weights=R['n_free'], minlength=m)
    s = np.bincount(R['user'], weights=R['sum_free'], minlength=m)
    op = np.full(m, np.nan)
    ok = n >= CFG['opinion_min_posts']
    op[ok] = s[ok] / n[ok]
    return op


def follow_table(sources):
    F = np.load(os.path.join(L.DERIVED, 'follows.npy'), mmap_mode='r')
    keep = np.isin(F[:, 0], sources)
    return np.asarray(F[keep])


def distance_cuts(Fs, op):
    d = np.abs(op[Fs[:, 0]] - op[Fs[:, 1]])
    d = d[np.isfinite(d)]
    return np.percentile(d, [100 / 3, 200 / 3])


def run_p3(half, US, UD, cand, U_all, out, nproc, cuts=None):
    t0, t1 = win('win_late')
    users = active_users(UD, cand, t0, t1)
    watch, non = roles(US, users)
    op = late_opinions(UD, t0, t1)
    op = np.concatenate([op, np.full(max(0, int(users.max()) + 1 - len(op)), np.nan)])
    F = follow_table(users)
    opj = np.full(len(F), np.nan)
    inr = F[:, 1] < len(op)
    opj[inr] = op[F[inr, 1]]
    Fo = F[np.isfinite(opj)]
    dist = np.abs(op[Fo[:, 0]] - op[Fo[:, 1]])
    if cuts is None:
        cuts = np.percentile(dist[np.isfinite(dist)], [100 / 3, 200 / 3])
    out['cuts'] = cuts
    src = Fo[:, 0]
    pos = np.searchsorted(users, src)
    n_op = np.bincount(pos, minlength=len(users))
    n_mid = np.bincount(pos, weights=(dist >= cuts[0]) & (dist < cuts[1]), minlength=len(users))
    outdeg = np.bincount(np.searchsorted(users, F[:, 0]), minlength=len(users))
    n_post, n_day = activity(UD, users, t0, t1)
    elig = non & (n_op >= CFG['p3_min_followees']) & np.isfinite(op[users])
    idx = np.flatnonzero(elig)
    share = n_mid[idx] / n_op[idx]
    Z = np.column_stack([np.ones(len(idx)), np.log1p(outdeg[idx]), np.log1p(n_op[idx]),
                         np.log1p(n_post[idx]), np.log1p(n_day[idx])])
    beta = np.linalg.lstsq(Z, share, rcond=None)[0]
    resid = share - Z @ beta
    X, _ = L.daily_opinion(UD, users[idx], CFG['field'], t0, t1)
    S = L.user_states(X, t0)
    U = U_all[t0:t1]
    rows = valid_rows(S, t0, t1)
    order = np.argsort(resid)
    groups = np.array_split(order, CFG['p3_groups'])
    pool = Pool(nproc, initializer=_init, initargs=({'S': S, 'U': U, 'rows': rows, 'n': len(idx)},))
    gmc = np.array(pool.map(_group_mc, groups))
    slope = _slope(gmc)
    perms = np.array(pool.map(_p3_perm, range(5000, 5000 + CFG['n_perm']))); pool.close()
    p = (1 + (perms <= slope).sum()) / (1 + len(perms))
    out['p3'] = dict(n=len(idx), rows=len(rows), group_mc=gmc, slope=slope, perms=perms, p=p,
                     share_mean=[share[g].mean() for g in groups],
                     resid_mean=[resid[g].mean() for g in groups],
                     n_post_mean=[n_post[idx][g].mean() for g in groups],
                     outdeg_mean=[outdeg[idx][g].mean() for g in groups])
    print(f'P3 n {len(idx)} rows {len(rows)} group MC {np.round(gmc, 3)} slope {slope:.4f} p {p:.4f}', flush=True)
    return users, op, cuts


def run_p4(half, US, UD, users, op, cuts, U_all, out):
    t0, t1 = win('win_late')
    I = np.load(os.path.join(L.DERIVED, 'interactions.npy'), mmap_mode='r')
    sel = (I['day'] >= t0 - 6) & (I['day'] < t1)
    I = np.asarray(I[sel])
    I = I[np.isin(I['src'], users)]
    F = follow_table(users)
    fk = np.sort(F[:, 0].astype(np.int64) << 32 | F[:, 1].astype(np.int64))
    ik = I['src'].astype(np.int64) << 32 | I['dst'].astype(np.int64)
    pos = np.searchsorted(fk, ik).clip(0, len(fk) - 1)
    I, ik = I[fk[pos] == ik], ik[fk[pos] == ik]
    # unique (edge, day)
    ed = np.unique(np.stack([ik, I['day'].astype(np.int64)], 1), axis=0)
    ek, day = ed[:, 0], ed[:, 1]
    new_ep = np.r_[True, (ek[1:] != ek[:-1]) | (day[1:] - day[:-1] > 7)]
    ep_id = np.cumsum(new_ep) - 1
    ep_start = day[new_ep]
    ep_end = np.maximum.reduceat(day, np.flatnonzero(new_ep))
    ep_edge = ek[new_ep]
    # live edges: >= 1 interaction inside the window
    inwin = day >= t0
    live_keys, n_int_days = np.unique(ek[inwin], return_counts=True)
    # switch events (day, edge)
    opens = ep_start >= t0 + 1
    closes = ep_end + 7 <= t1 - 1
    sw_edge = np.r_[ep_edge[opens], ep_edge[closes]]
    sw_day = np.r_[ep_start[opens], ep_end[closes] + 7]
    keep = np.isin(sw_edge, live_keys)
    sw_edge, sw_day = sw_edge[keep], sw_day[keep]
    e_idx = np.searchsorted(live_keys, sw_edge)
    n_sw = np.bincount(e_idx, minlength=len(live_keys))
    src = (live_keys >> 32).astype(np.int64); dst = (live_keys & 0xFFFFFFFF).astype(np.int64)
    opd = np.where(dst < len(op), op[np.minimum(dst, len(op) - 1)], np.nan)
    d = np.abs(op[src] - opd)
    ok = np.isfinite(d)
    cls = np.where(d < cuts[0], 0, np.where(d < cuts[1], 1, 2))
    strata = np.searchsorted(np.array(CFG['p4_strata']), n_int_days, side='right') - 1
    # news intensity
    U = U_all[t0:t1]
    nrm = np.linalg.norm(U, axis=1)
    days = np.arange(t0 + 1, t1)
    v = nrm[days - t0]
    fin = np.isfinite(v)
    lo_c, hi_c = np.nanpercentile(v, [100 * CFG['p4_news_frac'], 100 * (1 - CFG['p4_news_frac'])])
    hi_days = set(days[fin & (v >= hi_c)]); lo_days = set(days[fin & (v <= lo_c)])
    is_hi = np.isin(sw_day, list(hi_days)); is_lo = np.isin(sw_day, list(lo_days))
    n_hi = np.bincount(e_idx, weights=is_hi, minlength=len(live_keys))
    n_lo = np.bincount(e_idx, weights=is_lo, minlength=len(live_keys))
    E = dict(src=src[ok], cls=cls[ok], strata=strata[ok], n_sw=n_sw[ok], n_hi=n_hi[ok],
             n_lo=n_lo[ok], n_int=n_int_days[ok])
    nh, nl = len(hi_days), len(lo_days)

    # per-source aggregates: counts and switch sums by (class, stratum); news-day switches by class
    us, inv = np.unique(E['src'], return_inverse=True)
    ns = len(CFG['p4_strata'])
    cell = E['cls'] * ns + E['strata']
    A_cnt = np.zeros((len(us), 3 * ns)); A_sw = np.zeros((len(us), 3 * ns))
    np.add.at(A_cnt, (inv, cell), 1); np.add.at(A_sw, (inv, cell), E['n_sw'])
    B_cnt = np.zeros((len(us), 3)); B_hi = np.zeros((len(us), 3)); B_lo = np.zeros((len(us), 3))
    np.add.at(B_cnt, (inv, E['cls']), 1); np.add.at(B_hi, (inv, E['cls']), E['n_hi'])
    np.add.at(B_lo, (inv, E['cls']), E['n_lo'])

    def stats(w):
        cnt = (w @ A_cnt).reshape(3, ns); sw = (w @ A_sw).reshape(3, ns)
        res = {}
        for other, name in ((0, 'close'), (2, 'far')):
            ok_s = (cnt[1] > 0) & (cnt[other] > 0)
            diff = sw[1, ok_s] / cnt[1, ok_s] - sw[other, ok_s] / cnt[other, ok_s]
            res[name] = float((cnt[1, ok_s] * diff).sum() / cnt[1, ok_s].sum())
        c, h, l = w @ B_cnt, w @ B_hi, w @ B_lo
        lr = np.log((h / (c * nh)) / (l / (c * nl)))
        res['logratio'] = lr
        res['news'] = float(lr[1] - 0.5 * (lr[0] + lr[2]))
        return res

    obs = stats(np.ones(len(us)))
    rng = np.random.default_rng(SEED)
    boots = [stats(np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us)).astype(float))
             for _ in range(CFG['n_boot'])]

    def pval(key):
        x = np.array([bb[key] for bb in boots])
        return (1 + (x <= 0).sum()) / (1 + len(x)), np.percentile(x, [2.5, 97.5])
    p_close, ci_close = pval('close'); p_far, ci_far = pval('far'); p_news, ci_news = pval('news')
    raw = {}
    for g, name in enumerate(('close', 'middling', 'far')):
        mg = E['cls'] == g
        raw[name] = dict(n=int(mg.sum()), switches=float(E['n_sw'][mg].mean()),
                         int_days=float(E['n_int'][mg].mean()),
                         per_int_day=float(E['n_sw'][mg].sum() / E['n_int'][mg].sum()),
                         logratio=float(obs['logratio'][g]))
    out['p4'] = dict(n_edges=int(ok.sum()), n_sources=len(us), raw=raw,
                     d_close=obs['close'], ci_close=ci_close, p_close=p_close,
                     d_far=obs['far'], ci_far=ci_far, p_far=p_far,
                     d_news=obs['news'], ci_news=ci_news, p_news=p_news,
                     hi_days=nh, lo_days=nl)
    print(f"P4 edges {ok.sum()} sources {len(us)}  mid-close {obs['close']:.4f} p {p_close:.4f}  "
          f"mid-far {obs['far']:.4f} p {p_far:.4f}  news {obs['news']:.4f} p {p_news:.4f}", flush=True)
    for k, v in raw.items():
        print('   ', k, v)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['describe', 'run'])
    ap.add_argument('--half', choices=['discovery', 'confirm'], required=True)
    ap.add_argument('--nproc', type=int, default=10)
    ap.add_argument('--only', default='1234')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    half = 0 if args.half == 'discovery' else 1
    US, UD, cand = base_population(half)
    U_all, ev, _, _ = L.news_series()
    if args.cmd == 'describe':
        describe(US, UD, cand)
        return
    out = dict(cfg=CFG, half=args.half)
    t = time.time()
    if '1' in args.only or '2' in args.only:
        run_p1_p2(half, US, UD, cand, U_all, out, args.nproc)
    if '3' in args.only or '4' in args.only:
        cuts = None
        if half == 1 and os.path.exists(os.path.join(ROOT, 'results', 'real_cuts.json')):
            cuts = np.array(json.load(open(os.path.join(ROOT, 'results', 'real_cuts.json'))))
        users, op, cuts = run_p3(half, US, UD, cand, U_all, out, args.nproc, cuts)
        if half == 0:
            json.dump(list(map(float, cuts)), open(os.path.join(ROOT, 'results', 'real_cuts.json'), 'w'))
        if '4' in args.only:
            run_p4(half, US, UD, users, op, cuts, U_all, out)
    out['seconds'] = time.time() - t
    np.save(os.path.join(ROOT, 'results', f'real_{args.half}{args.tag}.npy'), out, allow_pickle=True)


def describe(US, UD, cand):
    print('English candidates in half:', len(cand))
    for name in ('win_full', 'win_late'):
        t0, t1 = win(name)
        for frac in (0.4, 0.5, 0.6, 0.8):
            CFG['act_frac'] = frac
            u = active_users(UD, cand, t0, t1)
            w, n = roles(US, u)
            print(f'{name} act_frac {frac}: {len(u)} users, {w.sum()} watchers, {n.sum()} non-watchers')
    CFG['act_frac'] = 0.6


if __name__ == '__main__':
    main()

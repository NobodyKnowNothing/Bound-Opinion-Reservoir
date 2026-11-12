"""Text summary and figure for a real-data run.

Usage: python scripts/real_report.py discovery|confirm [2]
Merges results/real_<half>[_r2]_p12.npy and _p34.npy; writes results/real_<half>[_r2].txt
and figures/real_<half>[_r2].png. The optional 2 selects round 2 (attention states).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import real_lib as L  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')


def main():
    half = sys.argv[1]
    rnd = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    tag = '' if rnd == 1 else '_r2'
    prereg = 'REAL_PREREG.md' if rnd == 1 else 'REAL_PREREG2.md'
    r = {}
    for part in ('_p12', '_p34'):
        f = os.path.join(ROOT, 'results', f'real_{half}{tag}{part}.npy')
        if os.path.exists(f):
            r.update(np.load(f, allow_pickle=True).item())
    lines = [f'Real-data test, round {rnd} '
             f'({"daily sentiment" if rnd == 1 else "topic attention"} states), {half} half', '']
    pv = {}
    if 'p1' in r:
        p1 = r['p1']
        lines += [f"P1 there is memory: users {r['p1_users']}, days {r['p1_rows']}",
                  f"  MC = {p1['mc']:.3f}; shifted-news null mean {p1['null'].mean():.3f}, "
                  f"95th pct {np.percentile(p1['null'], 95):.3f}, max {p1['null'].max():.3f} "
                  f"({len(p1['null'])} shifts); p = {p1['p']:.4f}",
                  '  per-lag increment: ' + ' '.join(f'{x:+.3f}' for x in p1['inc']),
                  '  per-lag baseline R2: ' + ' '.join(f'{x:.2f}' for x in p1['base']), '']
        pv['P1'] = p1['p']
    if 'p2' in r:
        p2 = r['p2']
        lines += [f"P2 second-hand users hold less: {p2['n_pairs']} matched pairs",
                  f"  activity (posts, active days) watchers {p2['act_watch'][0]:.0f}, {p2['act_watch'][1]:.0f}; "
                  f"non-watchers {p2['act_non'][0]:.0f}, {p2['act_non'][1]:.0f}",
                  f"  MC watchers {p2['mc_watch']:.3f}, non-watchers {p2['mc_non']:.3f}, "
                  f"difference {p2['diff']:+.3f}; pair-swap null 95th pct {np.percentile(p2['perms'], 95):+.3f}; "
                  f"p = {p2['p']:.4f}", '']
        pv['P2'] = p2['p']
    if 'p3' in r:
        p3 = r['p3']
        lines += [f"P3 edge-of-tolerance follows cost memory: {p3['n']} non-watchers, {p3['rows']} days, "
                  f"distance cut points {np.round(r['cuts'], 4)}",
                  '  groups by residual middling share (low -> high):',
                  '    middling share ' + ' '.join(f'{x:.3f}' for x in p3['share_mean']),
                  '    group MC       ' + ' '.join(f'{x:+.3f}' for x in p3['group_mc']),
                  '    mean posts     ' + ' '.join(f'{x:.0f}' for x in p3['n_post_mean']),
                  '    mean follows   ' + ' '.join(f'{x:.0f}' for x in p3['outdeg_mean']),
                  f"  slope {p3['slope']:+.4f} per group; random-group null 5th pct "
                  f"{np.percentile(p3['perms'], 5):+.4f}; p = {p3['p']:.4f}", '']
        pv['P3'] = p3['p']
    if 'p4' in r:
        p4 = r['p4']
        lines += [f"P4 middling links switch with the news: {p4['n_edges']} live follow edges from "
                  f"{p4['n_sources']} users; {p4['hi_days']} high-news and {p4['lo_days']} low-news days"]
        for k, v in p4['raw'].items():
            lines.append(f"    {k:9s} n {v['n']:7d}  switches/edge {v['switches']:.3f}  interaction days "
                         f"{v['int_days']:.2f}  switches per interaction day {v['per_int_day']:.3f}  "
                         f"log(high/low news rate) {v['logratio']:+.3f}")
        lines += [f"  4a middling - close (within interaction strata) {p4['d_close']:+.4f} "
                  f"[{p4['ci_close'][0]:+.4f}, {p4['ci_close'][1]:+.4f}] p = {p4['p_close']:.4f}",
                  f"  4a middling - far                               {p4['d_far']:+.4f} "
                  f"[{p4['ci_far'][0]:+.4f}, {p4['ci_far'][1]:+.4f}] p = {p4['p_far']:.4f}",
                  f"  4b news-day excess for middling                 {p4['d_news']:+.4f} "
                  f"[{p4['ci_news'][0]:+.4f}, {p4['ci_news'][1]:+.4f}] p = {p4['p_news']:.4f}", '']
        pv['P4a-close'] = p4['p_close']; pv['P4a-far'] = p4['p_far']; pv['P4b'] = p4['p_news']
    if pv:
        adj = dict(zip(pv, L.holm(list(pv.values()))))
        lines.append('One-sided p values, Holm-adjusted across the tests above:')
        for k, p in pv.items():
            lines.append(f'  {k:10s} p {p:.4f}  Holm {adj[k]:.4f}  {"PASS" if adj[k] < 0.05 else "fail"}')
        ok = {k: a < 0.05 for k, a in adj.items()}
        lines += ['', f'Verdict under the pre-registered rules ({prereg}):']
        if 'P1' in ok:
            lines.append(f"  1 there is memory: {'supported' if ok['P1'] else 'not supported'}")
            for k, name in (('P2', '2 second-hand users hold less'), ('P3', '3 middling follows cost memory')):
                if k in ok:
                    v = ('supported' if ok[k] else 'not supported') if ok['P1'] else                         'not interpretable (prediction 1 failed: the measure is too blunt)'
                    lines.append(f'  {name}: {v}')
        if 'P4a-close' in ok:
            lines.append(f"  4a middling links switch more than close and far links: "
                         f"{'supported' if ok['P4a-close'] and ok['P4a-far'] else 'not supported'}")
            lines.append(f"  4b more so on high-news days: {'supported' if ok['P4b'] else 'not supported'}")
    txt = '\n'.join(lines)
    print(txt)
    open(os.path.join(ROOT, 'results', f'real_{half}{tag}.txt'), 'w').write(txt + '\n')

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    if 'p1' in r:
        ax[0].hist(r['p1']['null'], bins=20, color='0.7', label='shifted news')
        ax[0].axvline(r['p1']['mc'], color='C3', lw=2, label='observed')
        ax[0].set(title='P1: memory capacity vs null', xlabel='MC (sum of incremental R²)')
        ax[0].legend(frameon=False)
    if 'p2' in r:
        ax[1].hist(r['p2']['perms'], bins=20, color='0.7', label='pair-swap null')
        ax[1].axvline(r['p2']['diff'], color='C3', lw=2, label='observed')
        ax[1].set(title='P2: MC watchers − non-watchers', xlabel='difference')
        ax[1].legend(frameon=False)
    if 'p3' in r:
        ax[2].plot(np.arange(1, len(r['p3']['group_mc']) + 1), r['p3']['group_mc'], 'o-', color='C0')
        ax[2].set(title='P3: MC by middling-follow share', xlabel='group (low → high share)',
                  ylabel='group MC')
    if 'p4' in r:
        names = list(r['p4']['raw'])
        ax[3].bar(names, [r['p4']['raw'][k]['per_int_day'] for k in names], color=['C2', 'C1', 'C0'])
        ax[3].set(title='P4: switches per interaction day', ylabel='switches / interaction day')
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'figures', f'real_{half}{tag}.png'), dpi=120)


if __name__ == '__main__':
    main()

"""Bluesky follow graph and interactions -> compact numpy arrays.

Usage: python -I scripts/real_extract_edges.py data/bluesky data/derived

follows.npy       (E, 2) int32, unique (follower, followee) pairs from followers.csv.gz
interactions.npy  structured (src, dst, day, kind) for replies (kind 0), reposts (1) and
                  quotes (2) from interactions.csv.gz; thread-root authors are not counted.
                  day = days since 2023-02-17; self-interactions dropped.
"""
import sys, os, gzip, time
import numpy as np

DAY0 = np.datetime64('2023-02-17')
CHUNK = 1 << 24
I_DTYPE = np.dtype([('src', 'i4'), ('dst', 'i4'), ('day', 'i2'), ('kind', 'i1')])


def parse_ints(buf, ncol):
    """Vectorized parse of comma/newline separated non-negative integers; 'None' -> -1."""
    a = np.frombuffer(buf.replace(b'None', b'N'), np.uint8)
    sep = (a == 44) | (a == 10)
    fid = np.cumsum(sep) - sep                      # field index of every byte
    ends = np.flatnonzero(sep)                      # position of the separator closing field f
    dig = (a >= 48) & (a <= 57)
    pos = np.flatnonzero(dig)
    p10 = ends[fid[pos]] - pos - 1
    vals = np.bincount(fid[pos], weights=(a[pos] - 48) * 10.0 ** p10, minlength=len(ends))
    vals[np.bincount(fid[a == 78], minlength=len(ends)) > 0] = -1
    return vals.reshape(-1, ncol)


def numeric_chunks(path, ncol):
    """Yield (n, ncol) float64 arrays; 'None' becomes -1."""
    rest = b''
    with gzip.open(path, 'rb') as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            buf = rest + buf
            cut = buf.rfind(b'\n') + 1
            buf, rest = buf[:cut], buf[cut:]
            yield parse_ints(buf, ncol)
    if rest.strip():
        yield parse_ints(rest.rstrip(b'\n') + b'\n', ncol)


def day_of(stamp):
    """YYYYMMDDHHMM (float) -> day index."""
    s = (stamp // 10000).astype(np.int64)
    y, m, d = s // 10000, s // 100 % 100, s % 100
    dates = (y - 1970).astype('datetime64[Y]') + (m - 1).astype('timedelta64[M]')
    dates = dates.astype('datetime64[D]') + (d - 1).astype('timedelta64[D]')
    return (dates - DAY0).astype(np.int64)


def main():
    src, out = sys.argv[1], sys.argv[2]
    t0 = time.time()
    parts = []
    done = os.path.exists(os.path.join(out, 'interactions.npy'))
    for a in ([] if done else numeric_chunks(os.path.join(src, 'interactions.csv.gz'), 6)):
        day = day_of(a[:, 5])
        for kind, col in ((0, 1), (1, 3), (2, 4)):
            ok = (a[:, col] >= 0) & (a[:, col] != a[:, 0]) & (day >= 0) & (day < 396)
            r = np.empty(ok.sum(), I_DTYPE)
            r['src'], r['dst'], r['day'], r['kind'] = a[ok, 0], a[ok, col], day[ok], kind
            parts.append(r)
        print(f'{time.time() - t0:.0f}s interactions {sum(map(len, parts)):,}', flush=True)
    if not done:
        I = np.concatenate(parts); del parts
        np.save(os.path.join(out, 'interactions.npy'), I)
        print('interactions', len(I), np.bincount(I['kind']), flush=True)
        del I
    # Keep only follows whose two ends have opinion rows (userdays.bin): those are the
    # only edges the analysis can use, and the full graph does not fit in memory deduplicated.
    sys.path.insert(0, os.path.dirname(__file__))
    from real_extract_posts import UD_DTYPE
    pr = os.path.join(out, 'userdays.bin')
    UD = np.memmap(pr, dtype=UD_DTYPE, mode='r', shape=(os.path.getsize(pr) // UD_DTYPE.itemsize,))
    have = np.unique(np.asarray(UD['user']))
    del UD
    parts = []
    n_all = 0
    for a in numeric_chunks(os.path.join(src, 'followers.csv.gz'), 2):
        a = a.astype(np.int64)
        n_all += len(a)
        a = a[np.isin(a[:, 0], have) & np.isin(a[:, 1], have)]
        parts.append(a[:, 0] << 32 | a[:, 1])
    key = np.unique(np.concatenate(parts)); del parts
    print(f'follow rows read {n_all:,}', flush=True)
    F = np.stack([key >> 32, key & 0xFFFFFFFF], 1).astype(np.int32)
    F = F[F[:, 0] != F[:, 1]]
    np.save(os.path.join(out, 'follows.npy'), F)
    print(f'{time.time() - t0:.0f}s follows {len(F):,}', flush=True)


if __name__ == '__main__':
    main()

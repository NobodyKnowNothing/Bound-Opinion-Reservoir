"""One pass over Bluesky user_posts.tar.gz -> compact per-user-day opinion aggregates.

Usage: python -I scripts/real_extract_posts.py data/bluesky/user_posts.tar.gz data/derived

Writes (all under the output dir, which lives in the git-ignored data/):
  users.npy     per-user summary for every user file in the archive
  userdays.npy  per-(user, day) aggregates for users with >= MIN_SENT English posts

Opinion of one post: s = (sent_label - 1) * sent_score, in [-1, 1]
(labels 0/1/2 = negative/neutral/positive from twitter-roberta; score = label confidence).
Only the user's own posts count (reposts are someone else's text). A post "mentions the
news" if it uses a word or word pair from the news lexicon (text_mask bit 0); the *_free columns leave those posts out, so decoded
memory reflects opinion, not people repeating headlines. Post text never leaves the
worker; only counts and sums are written.
"""
import sys, os, json, re, tarfile, subprocess, shutil, time
import numpy as np
from multiprocessing import Pool

DAY0 = np.datetime64('2023-02-17')
NDAYS = 396                     # 2023-02-17 .. 2024-03-18
MIN_SENT = 20                   # English sentiment posts needed to get user-day rows

NEWS_DOMAINS = r"""nytimes washingtonpost theguardian bbc cnn reuters apnews npr nbcnews cbsnews
abcnews go\.com politico axios bloomberg wsj ft\.com aljazeera thehill latimes usatoday foxnews
newsweek theatlantic vox motherjones huffpost msnbc independent\.co\.uk cnbc propublica nymag
thedailybeast salon slate newyorker time\.com forbes businessinsider france24 dw\.com euronews
cbc\.ca abc\.net\.au news\.yahoo theintercept rollingstone vanityfair talkingpointsmemo
rawstory mediaite newrepublic thenation jacobin semafor thedailybeast nbcnews sfchronicle
bostonglobe chicagotribune miamiherald seattletimes texastribune theverge arstechnica wired
techcrunch 404media cnet engadget nature\.com science\.org scientificamerican economist
telegraph\.co\.uk thetimes\.co\.uk skynews dailymail nypost washingtonexaminer
spiegel lemonde elpais haaretz timesofisrael kyivindependent pbs\.org""".split()
NEWS_LINK = re.compile(r'(?:https?://|www\.|\b)(?:[a-z0-9-]+\.)*(?:' + '|'.join(NEWS_DOMAINS) + r')[a-z.]*/', re.I)
ANY_LINK = re.compile(r'https?://|\b[a-z0-9-]+\.(?:com|org|net|co\.uk|news|gov)/', re.I)

_COUNTRIES = r"""afghanistan albania algeria argentina armenia australia austria azerbaijan bahrain
bangladesh belarus belgium bolivia bosnia brazil bulgaria burma cambodia cameroon canada chad
chile china colombia congo croatia cuba cyprus czech denmark ecuador egypt eritrea estonia
ethiopia finland france georgia germany ghana greece guatemala haiti honduras hungary iceland
india indonesia iran iraq ireland israel italy jamaica japan jordan kazakhstan kenya korea kosovo
kuwait kyrgyzstan laos latvia lebanon libya lithuania mali mexico moldova mongolia morocco
myanmar nepal netherlands nicaragua niger nigeria norway oman pakistan palestine panama
paraguay peru philippines poland portugal qatar romania russia rwanda saudi senegal serbia
singapore slovakia slovenia somalia spain sudan sweden switzerland syria taiwan tajikistan
tanzania thailand tunisia turkey uganda ukraine emirates uk britain england scotland wales
venezuela vietnam yemen zambia zimbabwe gaza kyiv kiev moscow beijing jerusalem tehran
crimea donbas bakhmut avdiivka rafah khan\ younis west\ bank washington"""
_DEMONYMS = r"""afghan algerian american argentinian armenian australian azerbaijani belarusian
brazilian british canadian chinese colombian cuban egyptian ethiopian european french german
greek haitian indian iranian iraqi irish israeli israelis italian japanese jordanian korean
lebanese libyan mexican nigerian pakistani palestinian palestinians polish russian russians
saudi serbian somali sudanese syrian taiwanese turkish ukrainian ukrainians venezuelan
yemeni"""
_ACTORS = r"""biden trump putin zelensky zelenskyy netanyahu xi\ jinping modi sunak macron scholz
erdogan kim\ jong lukashenko prigozhin wagner hamas hezbollah houthis houthi idf nato
united\ nations kremlin pentagon white\ house congress senate senator congressman
congresswoman house\ speaker speaker\ johnson mccarthy desantis haley pence harris obama
gop republicans? democrats? supreme\ court scotus parliament prime\ minister president
election elections primary primaries caucus ballot governor mayor minister ministry
fbi cia doj justice\ department indictment indicted arraignment trial verdict jury judge
lawsuit court sanctions tariffs? embassy diplomats? ambassador"""
_EVENTS = r"""war wars warfare invasion invaded troops military army soldiers airstrikes?
air\ strikes? missiles? rockets? drones? bombs? bombing bombed shelling ceasefire truce
hostages? genocide massacre terrorists? terrorism terror attack attacks attacked shooting
shootings shooter gunman killed killing deaths dead casualties wounded refugees evacuation
protests? protesters riots? coup earthquake hurricane typhoon cyclone wildfires? floods?
flooding tornado tsunami disaster derail(?:ed|ment) explosion crash outbreak pandemic
covid strike strikes striking union walkout layoffs inflation recession interest\ rates
fed\ rates economy stock\ market shutdown debt\ ceiling default bank\ collapse breaking
headline headlines reported according\ to"""


def _expand(term):
    """'republicans?' -> ['republican', 'republicans']; 'derail(?:ed|ment)' -> two words."""
    m = re.fullmatch(r'(.*?)\(\?:(.*)\)\??', term)
    if m:
        return [m.group(1) + x for x in m.group(2).split('|')]
    if term.endswith('?'):
        return [term[:-2], term[:-1]]
    return [term]


LEX_UNI, LEX_BI = set(), set()
for _t in (_COUNTRIES + '\n' + _DEMONYMS + '\n' + _ACTORS + '\n' + _EVENTS).replace('\\ ', '_').split():
    for _w in _expand(_t):
        _w = _w.replace('_', ' ')
        (LEX_BI if ' ' in _w else LEX_UNI).add(_w)
# Fixed topics for the stance proxy (topic-conditional sentiment); secondary use only.
# Each topic: (exact words, word prefixes, bigrams).
TOPICS = [
    ({'gaza', 'hamas', 'netanyahu', 'idf', 'rafah'}, ('israel', 'palestin'), {'west bank'}),
    ({'putin', 'kremlin', 'kyiv', 'crimea'}, ('ukrain', 'russia', 'zelensk'), set()),
    ({'maga', 'gop', 'desantis'}, ('trump', 'republican'), set()),
    ({'harris'}, ('biden', 'democrat'), {'white house'}),
    ({'ai', 'chatgpt', 'openai', 'llm', 'llms', 'gpt4'}, (), {'artificial intelligence', 'gpt 4'}),
    ({'musk', 'elon', 'twitter', 'tweet', 'tweets', 'threads', 'meta', 'zuckerberg'}, (), set())]
NT = len(TOPICS)
TOKEN = re.compile(r"[a-z0-9]+")
_CACHE = {}
_BI = {b: 1 for b in LEX_BI}
for _k, (_, _, _bis) in enumerate(TOPICS):
    for _b in _bis:
        _BI[_b] = _BI.get(_b, 0) | (2 << _k)
_BI_FIRST = {b.split()[0] for b in _BI}


def _tok_mask(w):
    m = 1 if w in LEX_UNI else 0
    for k, (ex, pre, _) in enumerate(TOPICS):
        if w in ex or (pre and w.startswith(pre)):
            m |= 2 << k
    return m


def text_mask(text):
    """bit 0: mentions the news lexicon; bit k+1: mentions topic k."""
    toks = TOKEN.findall(text.lower())
    m = 0
    for i, w in enumerate(toks):
        b = _CACHE.get(w)
        if b is None:
            b = _CACHE[w] = _tok_mask(w)
        m |= b
        if w in _BI_FIRST and i + 1 < len(toks):
            m |= _BI.get(w + ' ' + toks[i + 1], 0)
    return m


UD_DTYPE = np.dtype([('user', 'i4'), ('day', 'i2'), ('n_own', 'i2'), ('n_sent', 'i2'),
                     ('sum_s', 'f4'), ('n_free', 'i2'), ('sum_free', 'f4'), ('n_news', 'i2'),
                     ('n_link', 'i2'), ('n_reply', 'i2'), ('n_repost', 'i2'),
                     ('tn', 'i2', NT), ('ts', 'f4', NT)])
U_DTYPE = np.dtype([('user', 'i4'), ('n_posts', 'i4'), ('n_own', 'i4'), ('n_eng', 'i4'),
                    ('n_sent', 'i4'), ('n_free', 'i4'), ('n_news', 'i4'), ('n_link', 'i4'),
                    ('first', 'i2'), ('last', 'i2'), ('days_sent', 'i2'), ('weeks_sent', 'i2')])


def day_index(date):
    s = str(date)
    try:
        d = np.datetime64(f'{s[:4]}-{s[4:6]}-{s[6:8]}')
    except ValueError:
        return -1
    return int((d - DAY0).astype(int))


def process_user(raw):
    rows = {}
    n_posts = n_own = n_eng = n_sent = n_free = n_news = n_link = 0
    first, last = 10**6, -1
    uid = -1
    for line in raw.splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        uid = d.get('user_id', uid)
        n_posts += 1
        t = day_index(d.get('date'))
        if t < 0 or t >= NDAYS:
            continue
        r = rows.get(t)
        if r is None:
            r = rows[t] = [0, 0, 0.0, 0, 0.0, 0, 0, 0, 0, [0] * NT, [0.0] * NT]
        if d.get('reposted_author') is not None:
            r[8] += 1
            continue
        n_own += 1
        r[0] += 1
        first, last = min(first, t), max(last, t)
        text = d.get('text') or ''
        if d.get('replied_author') is not None:
            r[7] += 1
        if ANY_LINK.search(text):
            r[6] += 1; n_link += 1
            if NEWS_LINK.search(text):
                r[5] += 1; n_news += 1
        if d.get('langs') == ['eng']:
            n_eng += 1
        lab, sc = d.get('sent_label'), d.get('sent_score')
        if lab is None or sc is None or d.get('langs') != ['eng']:
            continue
        s = (int(lab) - 1) * float(sc)
        n_sent += 1
        r[1] += 1; r[2] += s
        mask = text_mask(text)
        if not mask & 1:
            r[3] += 1; r[4] += s; n_free += 1
        for k in range(NT):
            if mask >> (k + 1) & 1:
                r[9][k] += 1; r[10][k] += s
    days_sent = [t for t, r in rows.items() if r[1] > 0]
    summ = (uid, n_posts, n_own, n_eng, n_sent, n_free, n_news, n_link,
            first if last >= 0 else -1, last, len(days_sent), len({t // 7 for t in days_sent}))
    out = []
    if n_sent >= MIN_SENT:
        for t, r in rows.items():
            out.append((uid, t, min(r[0], 32767), min(r[1], 32767), r[2], min(r[3], 32767), r[4],
                        min(r[5], 32767), min(r[6], 32767), min(r[7], 32767), min(r[8], 32767),
                        tuple(r[9]), tuple(r[10])))
    return summ, out


def process_batch(batch):
    summs, rows = [], []
    for raw in batch:
        s, o = process_user(raw.decode('utf-8', errors='replace'))
        summs.append(s); rows.extend(o)
    return np.array(summs, dtype=U_DTYPE), np.array(rows, dtype=UD_DTYPE)


def batches(path, size=256, skip=0):
    """Yield lists of raw user files; the first `skip` user files are read past (resume)."""
    exe = os.environ.get('GZIP_EXE') or shutil.which('gzip')
    if exe:
        gz = subprocess.Popen([exe, '-dc', path], stdout=subprocess.PIPE, bufsize=1 << 22)
        tf = tarfile.open(fileobj=gz.stdout, mode='r|')
    else:
        gz, tf = None, tarfile.open(path, mode='r|gz')
    batch = []
    seen = 0
    for m in tf:
        if not m.isfile():
            continue
        seen += 1
        if seen <= skip:
            if seen % 200000 == 0:
                print(f'skipped {seen:,}', flush=True)
            continue
        batch.append(tf.extractfile(m).read())
        if len(batch) >= size:
            yield batch; batch = []
    if batch:
        yield batch
    if gz:
        gz.wait()


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    limit = int(os.environ.get('LIMIT_BATCHES', 0))
    # resume: keep complete records already written and skip that many user files
    pu, pr = os.path.join(out, 'users.bin'), os.path.join(out, 'userdays.bin')
    resume = os.environ.get('RESUME') == '1' and os.path.exists(pu)
    skip = os.path.getsize(pu) // U_DTYPE.itemsize if resume else 0
    if resume:
        for path, dt in ((pu, U_DTYPE), (pr, UD_DTYPE)):
            with open(path, 'r+b') as f:
                f.truncate(os.path.getsize(path) // dt.itemsize * dt.itemsize)
        print(f'resuming after {skip:,} users', flush=True)
    nu = nr = 0
    fu = open(pu, 'ab' if resume else 'wb')
    fr = open(pr, 'ab' if resume else 'wb')
    with Pool(int(os.environ.get('NPROC', 10))) as pool:
        for i, (u, r) in enumerate(pool.imap(process_batch, batches(src, skip=skip), chunksize=1)):
            fu.write(u.tobytes()); fr.write(r.tobytes()); nu += len(u); nr += len(r)
            if i % 200 == 0:
                print(f'{time.time() - t0:7.0f}s users {nu:,} userday rows {nr:,}', flush=True)
            if limit and i + 1 >= limit:
                pool.terminate(); break
    fu.close(); fr.close()
    # .bin files are raw records; read with np.fromfile(path, dtype=U_DTYPE / UD_DTYPE)
    print(f'done {time.time() - t0:.0f}s users {nu:,} userday rows {nr:,}', flush=True)


if __name__ == '__main__':
    main()

"""
Build data/results/item_clusters.csv: the resampling cluster of every analysed item, for the
cluster bootstrap of Topic_Modeling/18_cluster_bootstrap.py (triage 2.5).

  Reddit    discussion thread: a submission and the comments collected with it. The raw
            collection writes each submission immediately followed by its comments, so the
            thread of a comment is the last submission above it.
  Telegram  channel x ISO week of the message.
  Guardian  ISO week of publication.
A sensitivity version with daily instead of weekly blocks is written alongside
(column cluster_day; Reddit threads are unchanged).

Run once by the authors: it needs the raw Reddit and Telegram collections and the private
pseudonym map (available on request). The output contains only the published pseudonymous
item ids and anonymous cluster labels, so it can be redistributed.
"""
import os
import json
import pandas as pd

from config import BASE_DIR, DATA_PROCESSED, DATA_RAW, DATA_RESULTS

CORPUS = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
RAW_REDDIT = os.path.join(DATA_RAW, "reddit_comments_raccolti.csv")
RAW_TELEGRAM = os.path.join(DATA_RAW, "telegram_messaggi_raccolti.csv")
# private: pseudonym -> original Reddit id; never published (see .gitignore)
PSEUDONYM_MAP = os.path.join(BASE_DIR, "private", "reddit_pseudonym_map_RICOSTRUITA.json")
OUT = os.path.join(DATA_RESULTS, "item_clusters.csv")
PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]


def main():
    df = pd.read_csv(CORPUS, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))].dropna(subset=['testo_lemmatizzato'])
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    items = df.groupby(['parent', 'fonte']).agg(date=('data_originale_str', 'first')).reset_index()
    dt = pd.to_datetime(items['date'], utc=True, format='mixed')
    items['week'], items['day'] = dt.dt.strftime('%G-W%V'), dt.dt.strftime('%Y-%m-%d')

    # Reddit threads from the raw collection order
    raw = pd.read_csv(RAW_REDDIT, usecols=['id_originale'])
    is_post = raw['id_originale'].str[0] == '1'
    raw['thread'] = is_post.cumsum()
    assert raw['thread'].min() >= 1, "a comment precedes the first submission"
    pmap = json.load(open(PSEUDONYM_MAP, encoding="utf-8"))
    inv = {}
    for pseudo, orig in pmap.items():
        inv.setdefault(orig, []).append(pseudo)
    thread_of = {}
    for orig, th in zip(raw['id_originale'], raw['thread']):
        for pseudo in inv.get(orig, []):
            thread_of[pseudo] = th

    # Telegram channels from the raw collection
    tel = pd.read_csv(RAW_TELEGRAM, usecols=['message_id', 'chat_title'])
    channel_of = dict(zip(tel['message_id'].astype(str), tel['chat_title']))
    chan_label = {c: f"channel_{i + 1}" for i, c in enumerate(sorted(tel['chat_title'].unique()))}

    def cluster(r, block='week'):
        if r.fonte == 'Reddit_Commento':
            th = thread_of.get(r.parent)
            return None if th is None else f"reddit_thread_{th:04d}"
        if r.fonte == 'Telegram':
            ch = channel_of.get(r.parent)
            return None if ch is None else f"telegram_{chan_label[ch]}_{getattr(r, block)}"
        return f"guardian_{getattr(r, block)}"

    items['cluster'] = [cluster(r) for r in items.itertuples()]
    items['cluster_day'] = [cluster(r, 'day') for r in items.itertuples()]
    missing = items['cluster'].isna()
    print("items:", len(items), "| without cluster:", int(missing.sum()))
    print(items[~missing].groupby('fonte').agg(items=('parent', 'size'), clusters=('cluster', 'nunique')).to_string())
    sizes = items[~missing].groupby(['fonte', 'cluster']).size()
    print(sizes.groupby(level=0).describe()[['min', '50%', 'max']].to_string())
    print('daily blocks:', items.groupby('fonte').cluster_day.nunique().to_dict())
    items[['parent', 'fonte', 'cluster', 'cluster_day']].to_csv(OUT, index=False)
    print("channel labels:", chan_label)


if __name__ == "__main__":
    main()

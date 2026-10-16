"""
Canonical item order for every topic-model script.

LDA as implemented in gensim (online variational Bayes, Hoffman et al. 2010) updates
the model in chunks, so its output depends on the order in which documents are
passed to it (Agrawal, Fu & Menzies 2018, "order effects"). The scripts build items
with a pandas groupby on the item identifier, so the order is fixed by the ids --
and pseudonymising the Reddit ids before publication changed it.

To make the order an explicit, declared parameter rather than a side effect of the
identifiers, data/results/item_order_manifest.csv lists every item (pseudonymised
parent id, source) in the exact order used for the published results. Every
topic-model script passes its items through canonical_order() before fitting.
"""
import os
import numpy as np
import pandas as pd

from config import DATA_RESULTS

MANIFEST = os.path.join(DATA_RESULTS, "item_order_manifest.csv")


def canonical_order(items, keys=('parent', 'fonte')):
    """Return `items` sorted by the manifest rank; fails if any item is missing from it."""
    man = pd.read_csv(MANIFEST)
    keys = [k for k in keys if k in items.columns]
    rank = items[keys].merge(man.drop_duplicates(keys)[keys + ['rank']], on=keys, how='left')['rank']
    if rank.isna().any():
        raise ValueError(f"{int(rank.isna().sum())} items are not in {MANIFEST}")
    return items.iloc[np.argsort(rank.to_numpy(), kind='stable')].reset_index(drop=True)

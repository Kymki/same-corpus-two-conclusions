"""
Replace Reddit comment identifiers in the public replication package with salted
pseudonyms.

Why: Section 3.2 states that Reddit material is reported "without thread context
sufficient to locate a poster", and the Data Availability statement says that
identifiers for social-media excerpts are withheld from the public package. A
Reddit comment id resolves directly to the comment, and therefore to its author,
so publishing the raw ids is inconsistent with both statements.

Why not simply delete the column: `parent` / `id_originale` is the join key
across the whole pipeline (items, topic assignments, sentiment, validation).
Deleting it would break the end-to-end reproducibility promised in Section 3.7.
A keyed pseudonym preserves every join while removing resolvability.

Scheme: pseudonym = "r_" + sha256(salt || base_id)[:12], with any "_s<N>"
sentence suffix preserved. The salt lives OUTSIDE the repository; without it the
mapping cannot be recomputed. The reverse mapping is written next to the salt so
that identifiers can still be supplied to readers on request, as the paper
promises.

Idempotent: ids already carrying the "r_" prefix are left alone.

Usage:
    python pseudonymise_reddit_ids.py            # apply
    python pseudonymise_reddit_ids.py --check    # report only, change nothing
"""
import hashlib
import json
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO, "data", "results")
# Kept outside the repository on purpose.
PRIVATE_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
SALT_FILE = os.path.join(PRIVATE_DIR, "reddit_pseudonym_salt.txt")
MAP_FILE = os.path.join(PRIVATE_DIR, "reddit_pseudonym_map_NON_pubblicare.json")

REDDIT = "Reddit_Commento"
PREFIX = "r_"
SUFFIX = re.compile(r"^(?P<base>.+?)(?P<suf>_s\d+)?$")

# Published files: these are tracked by git and constitute the public package.
TARGETS = [
    ("items_en_meta.csv", "parent"),
    ("document_topics_items_k5.csv", "parent"),
    ("document_topics_items_k7.csv", "parent"),
    ("document_topics_items_k10.csv", "parent"),
    ("document_topics_items_k15.csv", "parent"),
    ("document_topics_items_k21.csv", "parent"),
    ("fig8_candidates.csv", "id_originale"),
    ("validation_annotation_TODO.csv", "id_originale"),
    ("validation_annotation_labeled.csv", "id_originale"),
]

# Local working files: NOT tracked by git, so they are never published. They are
# rewritten too, because several scripts join a tracked file against one of these
# on the identifier (02_sentiment_validation.py merges the annotations against
# the predictions; 09_symfilter_continuous_measures.py merges the corpus against
# items_en_meta.csv). Leaving them on raw ids would silently drop every Reddit
# row from those joins.
#
# The raw collections under data/raw/ keep their original identifiers: they are
# the source of truth and are not redistributed. Re-running preprocessing from
# them therefore reintroduces raw ids downstream, and this script must be run
# again afterwards.
WORKING = [
    (os.path.join(REPO, "data", "results", "sentiment_results_multilingua.csv"), "id_originale"),
    (os.path.join(REPO, "data", "processed", "dati_testuali_preproc_consolidati.csv"), "id_originale"),
]


def load_salt():
    if os.path.exists(SALT_FILE):
        with open(SALT_FILE, encoding="utf-8") as f:
            salt = f.read().strip()
        if salt:
            return salt, False
    salt = os.urandom(32).hex()
    os.makedirs(PRIVATE_DIR, exist_ok=True)
    with open(SALT_FILE, "w", encoding="utf-8") as f:
        f.write(salt + "\n")
    return salt, True


def pseudonym(salt, base):
    return PREFIX + hashlib.sha256((salt + base).encode("utf-8")).hexdigest()[:12]


def main():
    check_only = "--check" in sys.argv
    salt, fresh = load_salt()
    print(f"salt: {SALT_FILE}{'  (generato ora)' if fresh else '  (esistente, riusato)'}")
    if check_only:
        print("modalita' --check: nessun file verra' modificato\n")

    mapping, changed_files, total = {}, [], 0

    def process(path, col, label):
        nonlocal total
        if not os.path.exists(path):
            print(f"  SALTATO (assente): {label}")
            return
        df = pd.read_csv(path, low_memory=False)
        for needed in (col, "fonte"):
            if needed not in df.columns:
                print(f"  SALTATO (nessuna colonna {needed}): {label}")
                return

        ids = df.loc[df["fonte"].astype(str).eq(REDDIT), col].astype(str)
        todo = ids[~ids.str.startswith(PREFIX)]
        if todo.empty:
            print(f"  {label:<44} 0 da sostituire (gia' pseudonimizzato)")
            return

        new = {}
        for raw in todo.unique():
            m = SUFFIX.match(raw)
            base, suf = m.group("base"), m.group("suf") or ""
            if base not in mapping:
                mapping[base] = pseudonym(salt, base)
            new[raw] = mapping[base] + suf

        # collision guard: distinct originals must not share a pseudonym
        inv = {}
        for b, p in mapping.items():
            if p in inv and inv[p] != b:
                sys.exit(f"ERRORE: collisione di pseudonimo tra {inv[p]} e {b}")
            inv[p] = b

        if not check_only:
            df.loc[todo.index, col] = todo.map(new)
            df.to_csv(path, index=False)
        changed_files.append(label)
        total += len(todo)
        print(f"  {label:<44} {len(todo):>5} identificativi -> pseudonimi")

    print("\n-- file pubblicati (tracciati da git) --")
    for name, col in TARGETS:
        process(os.path.join(RESULTS, name), col, name)

    print("\n-- file di lavoro (non tracciati, mai pubblicati) --")
    for path, col in WORKING:
        process(path, col, os.path.basename(path))

    if check_only:
        print(f"\n{total} identificativi Reddit da sostituire in {len(changed_files)} file.")
        return

    if mapping:
        old = {}
        if os.path.exists(MAP_FILE):
            with open(MAP_FILE, encoding="utf-8") as f:
                old = json.load(f)
        old.update(mapping)
        with open(MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(old, f, indent=2, ensure_ascii=False)
        print(f"\nmappa inversa ({len(old)} voci): {MAP_FILE}")
    print(f"sostituiti {total} identificativi in {len(changed_files)} file.")


if __name__ == "__main__":
    main()

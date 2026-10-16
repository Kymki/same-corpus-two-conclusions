"""
Programmatic topic classification for Section 4.8: diplomatic/institutional, kinetic, other.

DRAFT for approval (Remo Pareschi, two blind coders) -- frozen before any affective
charge was computed or inspected for the models it will be applied to.

Why a rule rather than inspection. The order sensitivity of LDA (Agrawal, Fu & Menzies
2018) means every repetition of the 2.6 design produces different topics, so the
classification must be re-derived for every model. Hand-coding twenty models blind is
not practicable; a declared rule applied unchanged to every model is, and it cannot be
adapted to the result.

The criterion is the one stated in the submitted paper (Figure 4 script): a topic counts
as diplomatic/institutional only if its leading terms name an OFFICE, an INSTITUTION or
a HEAD OF STATE OR GOVERNMENT; a place name alone is not enough. Kinetic topics are those
whose leading terms name military action or its effects. Each lexicon below is an
explicit test on the entity type a term denotes; it is written from that definition,
in lemma form (the topic terms are spaCy lemmas, lowercased), and contains no place names.

Decision rule, applied to the N_TERMS leading terms of a topic:
  D = number of terms in OFFICE | INSTITUTION | HEAD_OF_STATE
  K = number of terms in KINETIC
  diplomatic/institutional  if D >= 1 and D > K
  kinetic                   if K >= 1 and K > D
  other                     otherwise (no hit, or a tie)

Declared alternatives, to be reported alongside the main rule (triage 2.4: "report under
reasonable alternatives"):
  ALT_PROCESS  diplomatic lexicon also includes diplomatic-process terms (talk, summit,
               negotiation, ceasefire...), which the stated criterion does not mention;
  ALT_TOP5     the rule applied to the 5 leading terms instead of 10.

Deliberately excluded as ambiguous between categories or between noun and adjective:
"general", "air", "fire", "front", "defence"/"defense" (also "defence minister"),
"kim" (common first name), "force" (also "air force", "armed forces"). Lemmas of two
characters or fewer ("eu", "un", "xi", "mp", "g7") cannot occur: preprocessing drops them.

Named-entity recognition was considered and not used: spaCy NER is unreliable on single
lowercased lemmas detached from context, which is what a topic's term list is.
"""

N_TERMS = 10

OFFICE = {
    "minister", "ministry", "president", "presidency", "presidential", "chancellor",
    "premier", "secretary", "ambassador", "envoy", "diplomat", "diplomatic",
    "spokesman", "spokeswoman", "spokesperson", "official", "governor", "mayor",
    "commissioner", "senator", "lawmaker",
}
INSTITUTION = {
    "nato", "kremlin", "parliament", "government", "commission", "council",
    "congress", "senate", "embassy", "court", "administration", "cabinet", "ministry",
    "pentagon", "rada", "duma", "iaea", "osce", "g20",
}
HEAD_OF_STATE = {
    # heads of state or government named in the 2025-26 coverage of the war
    "putin", "zelenskyy", "zelensky", "trump", "macron", "starmer", "merz",
    "scholz", "erdogan", "erdoğan", "orban", "orbán", "modi", "lukashenko", "meloni",
    "tusk", "sanchez", "netanyahu", "pezeshkian", "khamenei", "biden", "carney",
    "albanese", "stubb", "fico", "nawrocki", "duda",
}
KINETIC = {
    "missile", "strike", "airstrike", "drone", "attack", "shelling", "shell", "artillery",
    "bomb", "bombing", "rocket", "explosion", "explode", "blast", "kill", "injure",
    "wound", "casualty", "damage", "destroy", "shoot", "assault", "offensive",
    "troop", "soldier", "combat", "frontline", "battle", "fighting",
    "intercept", "brigade", "infantry", "tank",
}
PROCESS = {"talk", "summit", "negotiation", "negotiate", "ceasefire", "truce", "deal",
           "agreement", "peace", "sanction", "meeting", "visit"}

DIPLOMATIC = OFFICE | INSTITUTION | HEAD_OF_STATE


def categorise(top_terms, n_terms=N_TERMS, diplomatic=DIPLOMATIC, kinetic=KINETIC):
    """top_terms: comma-separated string or list, most probable first."""
    if isinstance(top_terms, str):
        top_terms = [t.strip() for t in top_terms.split(",")]
    terms = [t.lower() for t in top_terms[:n_terms]]
    d = sum(t in diplomatic for t in terms)
    k = sum(t in kinetic for t in terms)
    if d >= 1 and d > k:
        return "Diplomatic/institutional"
    if k >= 1 and k > d:
        return "Kinetic"
    return "Other"


VARIANTS = {
    "main": dict(),
    "ALT_PROCESS": dict(diplomatic=DIPLOMATIC | PROCESS),
    "ALT_TOP5": dict(n_terms=5),
}

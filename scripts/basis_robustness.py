"""
Basis robustness check (Section 4, footnote).

The paper uses a single fixed basis: all 26 WordNet noun supersenses, identical
across every domain. In any one world only some categories are activated (seven
in legal, ten in oncology), because no discovered feature projects onto the
categories that do not occur there; those simply remain empty.

This script confirms that restricting the basis to the occurring categories
(14 in legal, 16 in oncology) makes no difference: the active categories and
the seed-to-seed agreement of the per-token interpretation arrays are identical
under the full 26-category basis and under the restricted one. The empty
categories carry no discovered feature and therefore neither add nor remove
anything. This is reported as a secondary check; the paper's design uses the
full 26-category basis throughout.

For each condition we report, across five seeds:
  - the number and identity of ACTIVE supersense categories after projection
  - the seed-to-seed agreement of per-token interpretation arrays

Usage:
    python scripts/basis_robustness.py
Requires embeddings/ (see embeddings/README.md) and data/ token lists.
"""
import numpy as np, json
from scipy.stats import rankdata
from sklearn.cluster import KMeans

# Full WordNet noun supersense inventory (26)
WN26 = sorted([
    'noun.Tops','noun.act','noun.animal','noun.artifact',
    'noun.attribute','noun.body','noun.cognition','noun.communication',
    'noun.event','noun.feeling','noun.food','noun.group',
    'noun.location','noun.motive','noun.object','noun.person',
    'noun.phenomenon','noun.plant','noun.possession','noun.process',
    'noun.quantity','noun.relation','noun.shape','noun.state',
    'noun.substance','noun.time'
])
IDX26 = {s: j for j, s in enumerate(WN26)}

PATHS = {
    'legal': ('embeddings/news_legal_embeddings.npy', 'data/news_legal_units.json'),
    'oncology': ('embeddings/oncology_embeddings.npy', 'data/oncology_units.json'),
}
SEEDS = [42, 7, 123, 2024, 99]

def supersenses(u):
    l = u.get('l1', '')
    if isinstance(l, list):
        return l or []
    return l.split('|') if l else []

def load(dom):
    emb_path, unit_path = PATHS[dom]
    emb = np.load(emb_path).astype(np.float64)
    units = json.load(open(unit_path))
    tok_ss = [supersenses(u) for u in units]
    N = emb.shape[0]
    dom_SS = sorted({x for p in tok_ss for x in p})
    dom_IDX = {s: j for j, s in enumerate(dom_SS)}
    WN_dom = np.zeros((N, len(dom_SS)), dtype=np.int8)
    WN_fix = np.zeros((N, 26), dtype=np.int8)
    for i, parts in enumerate(tok_ss):
        for p in parts:
            if p in dom_IDX: WN_dom[i, dom_IDX[p]] = 1
            if p in IDX26:   WN_fix[i, IDX26[p]] = 1
    return emb, WN_dom, WN_fix, dom_SS

def rho(M, PI, PJ, CRN):
    if M.shape[1] == 0: return 0.0
    a = M[PI]; b = M[PJ]
    inter = (a & b).sum(1); union = (a | b).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        j = np.where(union > 0, inter / union, 0.0)
    if j.std() < 1e-12: return 0.0
    jr = rankdata(j); jr = jr - jr.mean()
    return float((jr / np.linalg.norm(jr)) @ CRN)

def discover(EMB, PI, PJ, CRN, seed, max_feat=40, eps=0.001):
    km = KMeans(n_clusters=200, random_state=seed, n_init=3).fit(EMB)
    pr = km.cluster_centers_
    pr /= (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-12)
    s = EMB @ pr.T
    Z = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-12)
    Bd = (Z > 1.5).astype(np.int8)
    cands = [Bd[:, c] for c in range(200) if Bd[:, c].sum() > 0]
    M = np.array(cands).T
    sel = []; cur = 0.0; rem = list(range(M.shape[1]))
    while rem and len(sel) < max_feat:
        bc, br = None, cur
        for c in rem:
            r = rho(M[:, sel + [c]], PI, PJ, CRN)
            if r > br: br, bc = r, c
        if bc is None: break
        sel.append(bc); rem.remove(bc)
        if br - cur < eps and len(sel) >= 3: break
        cur = br
    return M[:, sel]

def project(feat_mat, basis_mat):
    N, k = feat_mat.shape
    base = basis_mat.mean(0)
    result = np.zeros((N, basis_mat.shape[1]), dtype=np.int8)
    for j in range(k):
        on = feat_mat[:, j].astype(bool)
        if on.sum() == 0: continue
        diff = basis_mat[on].mean(0) - base
        top = int(diff.argmax())
        if diff[top] < 0.1: continue
        result[:, top] = np.maximum(result[:, top], feat_mat[:, j])
    return result

def jac_direct(A, B):
    inter = (A & B).sum(1); union = (A | B).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        jv = np.where(union > 0, inter / union, 1.0)
    return jv.mean()

def seed_agr(mats):
    vals = [jac_direct(mats[i], mats[j])
            for i in range(len(mats)) for j in range(i+1, len(mats))]
    return float(np.mean(vals)), float(np.std(vals))

def active_cats(proj_list, SS_list):
    cats = set()
    for P in proj_list:
        for j in range(P.shape[1]):
            if P[:, j].sum() > 0: cats.add(SS_list[j])
    return sorted(cats)

def run(dom):
    EMB, WN_dom, WN_fix, dom_SS = load(dom)
    N = EMB.shape[0]
    rng = np.random.default_rng(0)
    mp = min(15000, N*(N-1)//2)
    P = rng.integers(0, N, size=(mp*2, 2)); P = P[P[:,0] != P[:,1]][:mp]
    PI, PJ = P[:, 0], P[:, 1]
    COS = np.einsum('ij,ij->i', EMB[PI], EMB[PJ])
    CR = rankdata(COS); CR = CR - CR.mean(); CRN = CR / (np.linalg.norm(CR) + 1e-12)
    feats = [discover(EMB, PI, PJ, CRN, s) for s in SEEDS]
    proj_dom = [project(F, WN_dom) for F in feats]
    proj_fix = [project(F, WN_fix) for F in feats]
    da = active_cats(proj_dom, dom_SS)
    fa = active_cats(proj_fix, WN26)
    dm, ds = seed_agr(proj_dom)
    fm, fs = seed_agr(proj_fix)
    return {
        'n_tokens': int(N),
        'domain_basis_size': len(dom_SS),
        'fixed_basis_size': 26,
        'domain_active': da,
        'fixed_active': fa,
        'categories_added_by_fixing': sorted(set(fa) - set(da)),
        'agreement_domain_basis': {'mean': round(dm,3), 'sd': round(ds,3)},
        'agreement_fixed_basis': {'mean': round(fm,3), 'sd': round(fs,3)},
        'feature_counts_per_seed': [int(F.shape[1]) for F in feats],
    }

if __name__ == '__main__':
    out = {}
    for dom in ['legal', 'oncology']:
        print(f'Running {dom} ...', flush=True)
        out[dom] = run(dom)
        print(json.dumps(out[dom], indent=2))
    with open('results/basis_robustness.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('\nWritten to results/basis_robustness.json')

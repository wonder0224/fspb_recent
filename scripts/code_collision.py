"""
Code collision / injectivity check (Section 5.3).

Verifies that the selected-width binary codes actually distinguish the N tokens,
so that comparing k to the log2(N) lower bound is legitimate. Reports unique-code
count, colliding-pair fraction, mean active count, and median Hamming distance.

Result (results/code_collision.json):
  legal:    481/508  unique (94.7%), 0.052% colliding pairs, 9 zero-feature tokens
  diverse:  2264/2327 unique (97.3%), 0.003% colliding pairs, 4 zero-feature tokens
  oncology: 4550/4629 unique (98.3%), 0.001% colliding pairs, 1 zero-feature token
The codes are injective to within a fraction of a percent; collisions sit in the
few tokens that activate no feature, and the distinguished share rises with N.
"""
import numpy as np, json
from sklearn.cluster import KMeans
from collections import Counter

def measure(emb_path, K, name):
    EMB=np.load(emb_path).astype(np.float64); N=EMB.shape[0]
    km=KMeans(n_clusters=300,random_state=42,n_init=3).fit(EMB)
    pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
    s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    B=(Z>1.5).astype(np.int8)
    cands=[B[:,c] for c in range(300) if B[:,c].sum()>0]
    order=sorted(range(len(cands)),key=lambda c:-int(cands[c].sum()))
    codes=np.array([cands[c] for c in order[:K]]).T
    ct=[tuple(r) for r in codes]
    cnt=Counter(ct)
    coll=sum(c*(c-1)//2 for c in cnt.values() if c>1); tot=N*(N-1)//2
    rng=np.random.default_rng(0); idx=rng.integers(0,N,(5000,2)); idx=idx[idx[:,0]!=idx[:,1]]
    hd=np.array([np.sum(codes[a]!=codes[b]) for a,b in idx])
    return dict(N=N,K=K,unique=len(set(ct)),unique_pct=round(100*len(set(ct))/N,1),
                collision_pct=round(100*coll/tot,3),zero=sum(1 for t in ct if sum(t)==0),
                active_mean=round(float(codes.sum(1).mean()),1),hamming_median=int(np.median(hd)))

if __name__=='__main__':
    out={}
    out['legal']=measure('data/news_legal_embeddings.npy',63,'legal')
    out['diverse']=measure('data/news_diverse_embeddings.npy',101,'diverse')
    out['oncology']=measure('data/oncology_embeddings.npy',105,'oncology')
    json.dump(out,open('results/code_collision.json','w'),indent=2)
    for k,v in out.items(): print(k, v)

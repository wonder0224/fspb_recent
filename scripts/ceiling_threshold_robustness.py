"""
Ceiling threshold robustness (Section 5.2).

Re-measures the semantic ceiling under a graded meaningfulness criterion
(two groups count as meaningfully split when the Jaccard overlap of their label
sets falls below tau) and compares it against the canonical dominant-label
test. Same discovery pipeline as the canonical run; only the split-judgment
changes.

Result (results/ceiling_threshold_robustness.json):
  legal:    dominant 63, tau=0.3 -> 63, tau=0.5 -> 66   (range 62-66)
  oncology: dominant 105, tau=0.3 -> 98, tau=0.5 -> 98  (range 98-105)
The ceiling shifts by only a few features under either criterion, well inside
the seed-to-seed spread, confirming it is not an artifact of one scoring rule.
"""
import numpy as np, json
from sklearn.cluster import KMeans

def labels_legal(u):
    L=set(); l1=u.get('l1','')
    for p in (l1.split('|') if l1 else []): L.add('ss:'+p)
    for ax in ['court','actor','action','topic']:
        v=u.get(ax,'')
        if v: L.add(f'{ax}:{v}')
    return L
def labels_onco(u):
    L=set(); l1=u.get('l1',[])
    for p in (l1 if isinstance(l1,list) else []): L.add('ss:'+p)
    for ax in ['type','system','finding','topic']:
        v=u.get(ax,'')
        if v: L.add(f'{ax}:{v}')
    return L

def load(dom):
    if dom=='legal':
        EMB=np.load('/mnt/user-data/uploads/news_legal_embeddings.npy').astype(np.float64)
        U=json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
        return EMB,[labels_legal(u) for u in U]
    EMB=np.load('/mnt/user-data/uploads/oncology_embeddings.npy').astype(np.float64)
    U=json.load(open('/mnt/user-data/uploads/oncology_units.json'))
    return EMB,[labels_onco(u) for u in U]

def disc_candidates(EMB,k_pool=300,z=1.5,seed=42):
    km=KMeans(n_clusters=k_pool,random_state=seed,n_init=3).fit(EMB)
    pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
    s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
    B=(Z>z).astype(np.int8)
    return [B[:,c] for c in range(k_pool) if B[:,c].sum()>0]

def msplit(members,nf,TOK_LAB,mode,tau):
    on=[i for i in members if nf[i]>0]; off=[i for i in members if nf[i]==0]
    if not on or not off: return 0,False
    moved=min(len(on),len(off))
    def labset(g):
        c={}
        for i in g:
            for l in TOK_LAB[i]: c[l]=c.get(l,0)+1
        return c
    con=labset(on); coff=labset(off)
    if mode=='dominant':
        t_on=max(con,key=con.get) if con else None
        t_off=max(coff,key=coff.get) if coff else None
        m=(t_on!=t_off) and t_on is not None and t_off is not None
    else:
        sa=set(con); sb=set(coff)
        ov=len(sa&sb)/len(sa|sb) if (sa|sb) else 1.0
        m = ov<tau
    return moved,m

def ceiling(EMB,TOK_LAB,mode,tau,max_feat=120):
    cands=disc_candidates(EMB)
    N=EMB.shape[0]
    order=sorted(range(len(cands)),key=lambda c:-int(cands[c].sum()))
    arr=[[] for _ in range(N)]
    sig_inc=[]
    for c in order[:max_feat]:
        feat=cands[c]
        keys=[tuple(a) for a in arr]
        classes={}
        for i,k in enumerate(keys): classes.setdefault(k,[]).append(i)
        step_sig=0
        for k,mem in classes.items():
            if len(mem)<2: continue
            mv,sig=msplit(mem,feat,TOK_LAB,mode,tau)
            if sig: step_sig+=mv
        for i in range(N): arr[i].append(int(feat[i]))
        sig_inc.append(step_sig)
    sig=np.array(sig_inc); window=10; ceil=None
    for s in range(len(sig)-window):
        if sig[s:s+window].mean()<0.5: ceil=s+1; break
    return ceil if ceil else len(sig)

def run(dom):
    EMB,LAB=load(dom)
    print(f'\n=== {dom} (N={EMB.shape[0]}) ===',flush=True)
    d=ceiling(EMB,LAB,'dominant',None)
    print(f'  dominant: {d}',flush=True)
    t3=ceiling(EMB,LAB,'graded',0.3)
    print(f'  τ=0.3: {t3}',flush=True)
    t5=ceiling(EMB,LAB,'graded',0.5)
    print(f'  τ=0.5: {t5}',flush=True)
    return {'dominant':d,'tau0.3':t3,'tau0.5':t5}

out={}
out['legal']=run('legal')
out['oncology']=run('oncology')
json.dump(out,open('/home/claude/ceiling_tau_result.json','w'),indent=2)
print('\n요약:',json.dumps(out))

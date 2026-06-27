"""
P4 형태-매칭 baseline (권고 6).
기존 random baseline은 크기만 맞춤 → named-entity/대문자/길이 confound 가능.
형태-매칭 baseline: 각 분기 집합과 같은 (길이·단어수·대문자시작) 분포를 갖는
무작위 집합을 뽑아 label overlap 비교. 분기가 여전히 높으면 label overlap이
형태가 아닌 의미를 잡는다는 증거.
"""
import numpy as np, json
from sklearn.cluster import KMeans
from collections import Counter

EMB=np.load('/mnt/user-data/uploads/news_legal_embeddings.npy').astype(np.float64)
U=json.load(open('/mnt/user-data/uploads/news_legal_units.json'))
N=EMB.shape[0]

def labels_of(u):
    L=set(); l1=u.get('l1','')
    for p in (l1.split('|') if l1 else []): L.add('ss:'+p)
    for ax in ['court','actor','action','topic']:
        v=u.get(ax,'')
        if v: L.add(f'{ax}:{v}')
    return L
TOK_LAB=[labels_of(u) for u in U]

# 형태 특성 벡터 (길이 구간, 단어수 구간, 대문자시작)
def morph(u):
    t=u.get('text','')
    nchar=len(t); nword=len(t.split()); cap=1 if t[:1].isupper() else 0
    # 구간화
    lb = 0 if nchar<12 else (1 if nchar<25 else 2)   # 짧/중/긴
    wb = 0 if nword<=1 else (1 if nword<=3 else 2)
    return (lb, wb, cap)
MORPH=[morph(u) for u in U]
# 형태 버킷별 토큰 인덱스
from collections import defaultdict
bucket=defaultdict(list)
for i,m in enumerate(MORPH): bucket[m].append(i)

def label_coherence(idx):
    if len(idx)<2: return None
    js=[]
    for a in range(len(idx)):
        for b in range(a+1,len(idx)):
            A,B=TOK_LAB[idx[a]],TOK_LAB[idx[b]]
            u=len(A|B); js.append(len(A&B)/u if u else 0)
    return float(np.mean(js))

rng=np.random.default_rng(0)
def morph_matched_baseline(idx, reps=50):
    """idx와 같은 형태 분포를 갖는 무작위 집합의 label coherence."""
    target=Counter(MORPH[i] for i in idx)
    vals=[]
    for _ in range(reps):
        pick=[]
        ok=True
        for m,cnt in target.items():
            pool=bucket[m]
            if len(pool)<cnt: ok=False; break
            pick += list(rng.choice(pool,size=cnt,replace=False))
        if ok and len(pick)>=2:
            v=label_coherence(pick)
            if v is not None: vals.append(v)
    return np.mean(vals) if vals else None

def plain_baseline(size, reps=50):
    vals=[]
    for _ in range(reps):
        idx=rng.choice(N,size=size,replace=False).tolist()
        v=label_coherence(idx)
        if v is not None: vals.append(v)
    return np.mean(vals) if vals else None

# 분기 트리 재구성 (naturalclass와 동일)
km=KMeans(n_clusters=200,random_state=42,n_init=3).fit(EMB)
pr=km.cluster_centers_; pr/=(np.linalg.norm(pr,axis=1,keepdims=True)+1e-12)
s=EMB@pr.T; Z=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-12)
Bd=(Z>1.5).astype(np.int8)
active=[c for c in range(200) if 0<int(Bd[:,c].sum())<N]
order=sorted(active,key=lambda c:-min(int(Bd[:,c].sum()),N-int(Bd[:,c].sum())))

branch_lab=[]; plain_rand=[]; morph_rand=[]
import sys; sys.setrecursionlimit(10000)
def split(idx, fpos):
    if fpos>=len(order) or len(idx)<3: return
    c=order[fpos]
    on=[i for i in idx if Bd[i,c]>0]; off=[i for i in idx if Bd[i,c]==0]
    if not on or not off: split(idx,fpos+1); return
    for child in [on,off]:
        if len(child)>=3:
            lc=label_coherence(child)
            if lc is not None:
                pr_b=plain_baseline(len(child))
                mr_b=morph_matched_baseline(child)
                if pr_b is not None and mr_b is not None:
                    branch_lab.append(lc); plain_rand.append(pr_b); morph_rand.append(mr_b)
    split(on,fpos+1); split(off,fpos+1)

split(list(range(N)),0)
branch_lab=np.array(branch_lab); plain_rand=np.array(plain_rand); morph_rand=np.array(morph_rand)
print(f"분기 자식집합(≥3) {len(branch_lab)}개")
print(f"\n라벨 일관성(label coherence):")
print(f"  분기:            {branch_lab.mean():.3f}")
print(f"  무작위(크기만):   {plain_rand.mean():.3f}  → 분기/무작위 = {branch_lab.mean()/plain_rand.mean():.2f}×")
print(f"  무작위(형태매칭): {morph_rand.mean():.3f}  → 분기/형태매칭 = {branch_lab.mean()/morph_rand.mean():.2f}×")
from scipy.stats import wilcoxon
_,p_plain=wilcoxon(branch_lab,plain_rand)
_,p_morph=wilcoxon(branch_lab,morph_rand)
print(f"\nWilcoxon p: 분기>크기만무작위 {p_plain:.2e}, 분기>형태매칭무작위 {p_morph:.2e}")
print(f"\n해석: 형태매칭 후에도 분기가 유의하게 높으면, label overlap은 형태 confound가 아닌 의미를 잡음")

json.dump(dict(n=len(branch_lab),
    branch=float(branch_lab.mean()),plain_rand=float(plain_rand.mean()),
    morph_rand=float(morph_rand.mean()),
    ratio_plain=float(branch_lab.mean()/plain_rand.mean()),
    ratio_morph=float(branch_lab.mean()/morph_rand.mean()),
    p_plain=float(p_plain),p_morph=float(p_morph)),
    open('/home/claude/p4_morph_matched_result.json','w'),indent=2)
print("[saved]")

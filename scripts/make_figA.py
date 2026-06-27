"""figA 아키텍처 흐름도 — Type 42 폰트로 재생성. 본문 P1 표현과 일관."""
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams["pdf.fonttype"]=42; mpl.rcParams["ps.fonttype"]=42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig,ax=plt.subplots(figsize=(13,5.6))
ax.set_xlim(0,13); ax.set_ylim(0,5.6); ax.axis('off')

# 상단 파이프라인 4박스
top=[
    ("Closed world","finite token set $W$\n(legal / oncology / news)","#D6E4F0","#2E5A88"),
    ("Floating-point\nembedding","$e_t\\in\\mathbb{R}^d$  (3 models)","#D6E4F0","#2E5A88"),
    ("Prototype induction","$k$ prototypes from\nrecurring regions\n(no labels)","#D8EAD8","#3E7D3E"),
    ("Binary feature array","$b_{ti}=\\mathbb{1}[z_{ti}\\geq\\tau]$\nper-token standardize + threshold","#D8EAD8","#3E7D3E"),
]
bw,bh,y0=2.7,1.3,4.0
xs=[0.3,3.4,6.5,9.6]
centers=[]
for (title,sub,fc,ec),x in zip(top,xs):
    box=FancyBboxPatch((x,y0),bw,bh,boxstyle="round,pad=0.05,rounding_size=0.12",
                       fc=fc,ec=ec,lw=1.6,zorder=2)
    ax.add_patch(box)
    ax.text(x+bw/2,y0+bh-0.32,title,ha='center',va='top',fontsize=11,fontweight='bold',color=ec,zorder=3)
    ax.text(x+bw/2,y0+0.42,sub,ha='center',va='center',fontsize=8.2,color='#333',zorder=3)
    centers.append(x+bw/2)
# 가로 화살표
for i in range(3):
    ax.add_patch(FancyArrowPatch((xs[i]+bw,y0+bh/2),(xs[i+1],y0+bh/2),
                 arrowstyle='-|>',mutation_scale=16,lw=1.6,color='#666',zorder=1))

# 중간 텍스트
ax.text(11.0,3.5,"the array is put to four refutable tests",ha='center',va='center',
        fontsize=9.5,style='italic',color='#555')
# 세로 분기 화살표 (이진배열 → 아래)
ax.add_patch(FancyArrowPatch((centers[3],y0),(centers[3],3.0),
             arrowstyle='-|>',mutation_scale=16,lw=1.6,color='#666'))
# 가로 분배선
ax.plot([1.6,11.0],[2.7,2.7],'-',color='#666',lw=1.4,zorder=1)
ax.add_patch(FancyArrowPatch((centers[3],3.0),(centers[3],2.7),arrowstyle='-',lw=1.4,color='#666'))

# 하단 P1~P4 (순서: P1 P3 P4 P2)
pcards=[
    ("P1  bounded ceiling","a finite width that stays\nin a narrow band ($\\approx$100)\nas the world grows"),
    ("P3  info-theoretic rate","width within a constant\nmultiple of $\\log_2 N$\n(not linear)"),
    ("P4  interpretability","label-free coordinates\nread as natural classes\n(person, place, ...)"),
    ("P2  binary vector","the form itself:\nordered binary array\n(constructive)"),
]
pbw,pbh,py=2.7,1.5,0.6
pxs=[0.3,3.4,6.5,9.6]
for (title,sub),x in zip(pcards,pxs):
    box=FancyBboxPatch((x,py),pbw,pbh,boxstyle="round,pad=0.05,rounding_size=0.12",
                       fc="#FBF0D8",ec="#B8860B",lw=1.5,zorder=2)
    ax.add_patch(box)
    ax.text(x+pbw/2,py+pbh-0.3,title,ha='center',va='top',fontsize=10.5,fontweight='bold',color="#8A6D0B",zorder=3)
    ax.text(x+pbw/2,py+0.5,sub,ha='center',va='center',fontsize=8.0,color='#333',zorder=3)
    # 분배선에서 각 카드로 화살표
    ax.add_patch(FancyArrowPatch((x+pbw/2,2.7),(x+pbw/2,py+pbh),
                 arrowstyle='-|>',mutation_scale=13,lw=1.3,color='#888',zorder=1))

plt.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01)
plt.savefig('/home/claude/tex_build/official/figA_architecture.pdf',bbox_inches='tight',pad_inches=0.1)
plt.close()
print("figA 재생성 (Type 42)")

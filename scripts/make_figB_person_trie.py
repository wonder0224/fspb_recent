#!/usr/bin/env python3
"""Figure B — 실제 person 트라이의 계층적 이진 분기 트리."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = json.load(open('/home/claude/person_branch_raw.json'))
nodes = {n['path']: n for n in d['nodes']}
MAXDEPTH = 5

role_color = {'attorney':'#4E8367','judge':'#C9A227','defendant':'#8E5BA6',
    'prosecutor':'#C25B4E','officer':'#3E8E8E','justice':'#B8860B',
    'witness':'#7A8B99','jury':'#A0522D','plaintiff':'#5B8DA6',
    'counsel':'#6B8E23','board':'#999999'}
def rcolor(a): return role_color.get(a, '#999999')

def node_actor(n):
    if n['kind']=='leaf': return n.get('actor','?'), n.get('actor_pur',1.0)
    a = n['on_actor'] if n['on']>=n['off'] else n['off_actor']
    return a, max(n['on_pur'],n['off_pur'])

def children(path):
    return [path+b for b in ('1','0') if path+b in nodes]

leaf_y=[0.0]; positions={}
def layout(path, depth):
    kids = children(path) if depth<MAXDEPTH else []
    if not kids:
        y=leaf_y[0]; leaf_y[0]+=1.0; positions[path]=(depth,y); return y
    ys=[layout(k,depth+1) for k in kids]
    y=float(np.mean(ys)); positions[path]=(depth,y); return y
layout('',0)

fig,ax=plt.subplots(figsize=(9.6,6.8)); XS=2.7
for path,(x,y) in positions.items():
    for k in children(path):
        if k in positions:
            xk,yk=positions[k]
            ax.plot([x*XS,xk*XS],[y,yk],'-',color='#B8C0C8',lw=1.3,zorder=1)

for path,(x,y) in positions.items():
    n=nodes[path]; size=n['size']; actor,pur=node_actor(n); col=rcolor(actor)
    r=4+1.7*np.sqrt(size)
    ax.scatter([x*XS],[y],s=r*r*3.0,c=col,alpha=0.22+0.55*pur,
               edgecolors=col,linewidths=1.3,zorder=2)

# 말단 라벨
for path,(x,y) in positions.items():
    n=nodes[path]; kids=children(path)
    isleaf=(x>=MAXDEPTH) or not kids
    if not isleaf: continue
    actor,pur=node_actor(n); col=rcolor(actor)
    if n['kind']=='leaf' and n.get('members'):
        nm=n['members'][0]
        if len(nm)>22: nm=nm[:20]+'…'
        lbl=f'{actor}: {nm}' if pur>=0.6 else nm
    else:
        lbl=actor if pur>=0.5 else 'mixed'
    ax.text(x*XS+1.1,y,lbl,ha='left',va='center',fontsize=7.3,color=col,
            fontweight='bold' if pur>=0.8 else 'normal',zorder=4)

rx,ry=positions['']
ax.text(rx*XS-0.7,ry,'noun.person\n(139 tokens)',ha='right',va='center',
        fontsize=10.5,fontweight='bold',color='#1F4E6B',zorder=4)

ymax=leaf_y[0]
ax.annotate("",xy=(MAXDEPTH*XS,ymax+0.4),xytext=(0,ymax+0.4),
            arrowprops=dict(arrowstyle='->',color='#555555',lw=1.4))
ax.text(0,ymax+1.0,'shallow splits\n(broad roles)',ha='left',va='bottom',fontsize=8.5,color='#555555')
ax.text(MAXDEPTH*XS,ymax+1.0,'deeper splits\n(purer → single referents)',ha='right',va='bottom',fontsize=8.5,color='#555555')

ax.set_xlim(-4.0,MAXDEPTH*XS+8); ax.set_ylim(-1,ymax+2.0); ax.axis('off')
plt.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01)
plt.savefig('/home/claude/tex_build/official/figB_person_nested.pdf',bbox_inches='tight',pad_inches=0.06)
plt.savefig('/home/claude/figB_tree_preview.png',dpi=130,bbox_inches='tight',pad_inches=0.06)
print("저장. 리프:",int(leaf_y[0]))

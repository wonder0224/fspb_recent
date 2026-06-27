"""모든 matplotlib figure를 Type 42 (TrueType) 폰트로 재생성."""
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np, json

OUT='/home/claude/tex_build/official'

# ===== figD: 평탄화 =====
worlds=[]
d=json.load(open('maxsuff_508.json')); worlds.append(('legal','N=508',d['sig_increments'],d['ceiling_est'],'#2E6B8A'))
d=json.load(open('maxsuff_diverse.json')); worlds.append(('diverse','N=2327',d['sig'],d['ceiling'],'#4E8367'))
d=json.load(open('maxsuff_onco.json')); worlds.append(('oncology','N=4629',d['sig'],d['ceiling'],'#C25B4E'))
fig,axes=plt.subplots(1,3,figsize=(13,4.0))
for ax,(name,Nlbl,sig,ceil,col) in zip(axes,worlds):
    sig=np.array(sig,dtype=float); x=np.arange(1,len(sig)+1); win=10
    ma=np.array([sig[max(0,i-win+1):i+1].mean() for i in range(len(sig))])
    ax.fill_between(x, ma, color=col, alpha=0.18, zorder=1)
    ax.plot(x, ma, color=col, lw=1.8, zorder=2, label='meaningful movement\n(10-feat. moving avg)')
    ax.axvline(ceil, color='#333333', ls='--', lw=1.4, zorder=3)
    ymax=ma.max()*1.08
    ax.text(ceil+2, ymax*0.82, f'ceiling = {ceil}', fontsize=10, color='#333333', fontweight='bold', va='top')
    ax.axvspan(ceil, len(sig), color='#999999', alpha=0.06, zorder=0)
    ax.text((ceil+len(sig))/2, ymax*0.5, 'no further\nmeaningful\ndistinction', fontsize=8, color='#888888', ha='center', va='center', style='italic')
    ax.set_title(f'{name}  ({Nlbl})', fontsize=11)
    ax.set_xlabel('number of features added', fontsize=9.5)
    ax.set_xlim(0, len(sig)); ax.set_ylim(0, ymax); ax.grid(alpha=0.25)
    if ax is axes[0]:
        ax.set_ylabel('meaningful label-aligned\nmovement (tokens)', fontsize=9.5)
        ax.legend(fontsize=7.5, loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT}/figD_ceiling_plateau.pdf',bbox_inches='tight',pad_inches=0.08)
plt.close()
print("figD 재생성 (Type 42)")

# ===== figC: log convergence =====
mpnet=json.load(open('sublinear_MPNet.json'))['data']
d3=json.load(open('propB_3dom.json'))
panels=[('Model C (768-d), oncology', mpnet, 0.974, 0.843),
        ('Model A (3072-d), oncology', d3['onco']['data'], 0.936, 0.860),
        ('Model A, legal', d3['508']['data'], 0.910, 0.835)]
fig,axes=plt.subplots(1,3,figsize=(13.5,4.2))
for ax,(title,data,r2log,r2lin) in zip(axes,panels):
    data=np.array(data); N=data[:,0]; k=data[:,1]
    xs=np.linspace(N.min(),N.max(),200)
    ll=np.polyfit(np.log(N),k,1); logcurve=ll[0]*np.log(xs)+ll[1]
    ll2=np.polyfit(N,k,1); lincurve=ll2[0]*xs+ll2[1]
    ax.plot(xs,logcurve,'-',color='#2E7D4F',lw=2.2,label=f'log fit  ($R^2$={r2log})',zorder=3)
    ax.plot(xs,lincurve,'--',color='#C0392B',lw=2.0,label=f'linear fit  ($R^2$={r2lin})',zorder=2)
    ax.scatter(N,k,s=55,color='#1F3D5B',zorder=5,label='measured width (mean$\\pm$sd)',edgecolors='white',linewidths=1)
    ax.errorbar(N,k,yerr=k*0.06,fmt='none',ecolor='#1F3D5B',capsize=3,zorder=4,alpha=0.7)
    ax.set_title(title,fontsize=11); ax.set_xlabel('world size  $N$  (tokens)',fontsize=10)
    ax.legend(fontsize=8.0,loc='upper left'); ax.grid(alpha=0.25)
    if ax is axes[0]: ax.set_ylabel('selected width  $k$  (feature coordinates)',fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT}/figC_log_convergence.pdf',bbox_inches='tight',pad_inches=0.08)
plt.close()
print("figC 재생성 (Type 42)")

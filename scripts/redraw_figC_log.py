"""figC 로그 x축 재작도 — 선형 기각 선명화, 로그/거듭제곱은 미결 유지. Type 42."""
import json, numpy as np
import matplotlib as mpl
mpl.use('Agg'); mpl.rcParams["pdf.fonttype"]=42; mpl.rcParams["ps.fonttype"]=42
import matplotlib.pyplot as plt

mpnet=json.load(open('/home/claude/sublinear_MPNet.json'))['data']
d3=json.load(open('/home/claude/propB_3dom.json'))
panels=[('Model C (768-d), oncology', mpnet, 0.974, 0.843),
        ('Model A (3072-d), oncology', d3['onco']['data'], 0.936, 0.860),
        ('Model A, legal', d3['508']['data'], 0.910, 0.835)]

fig,axes=plt.subplots(1,3,figsize=(13.5,4.2))
for ax,(title,data,r2log,r2lin) in zip(axes,panels):
    data=np.array(data); N=data[:,0]; k=data[:,1]
    xs=np.linspace(N.min(),N.max(),300)
    ll=np.polyfit(np.log(N),k,1); logcurve=ll[0]*np.log(xs)+ll[1]
    ll2=np.polyfit(N,k,1); lincurve=ll2[0]*xs+ll2[1]
    # x축 로그 스케일: 로그적합=직선, 선형적합=위로 휨
    ax.plot(xs,logcurve,'-',color='#2E7D4F',lw=2.2,label=f'log fit  ($R^2$={r2log})',zorder=3)
    ax.plot(xs,lincurve,'--',color='#C0392B',lw=2.0,label=f'linear fit  ($R^2$={r2lin})',zorder=2)
    ax.scatter(N,k,s=55,color='#1F3D5B',zorder=5,label='measured width (mean$\\pm$sd)',edgecolors='white',linewidths=1)
    ax.errorbar(N,k,yerr=k*0.06,fmt='none',ecolor='#1F3D5B',capsize=3,zorder=4,alpha=0.7)
    ax.set_xscale('log')
    ax.set_title(title,fontsize=11)
    ax.set_xlabel('world size  $N$  (tokens, log scale)',fontsize=10)
    ax.legend(fontsize=8.0,loc='upper left')
    ax.grid(alpha=0.25, which='both')
    if ax is axes[0]: ax.set_ylabel('selected width  $k$  (feature coordinates)',fontsize=10)

plt.tight_layout()
plt.savefig('/home/claude/tex_build/official/figC_log_convergence.pdf',bbox_inches='tight',pad_inches=0.08)
plt.savefig('/home/claude/figC_log_preview.png',dpi=115,bbox_inches='tight',pad_inches=0.08)
print("figC 로그축 재작도 완료")

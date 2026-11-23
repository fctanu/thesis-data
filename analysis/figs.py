import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from decimal import Decimal, ROUND_HALF_UP
def r1(x): return str(Decimal(str(round(x,6))).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP))
a=json.load(open('analysis/analysis.json')); C=['D0','D1','D2','D3','D4','D5','D6','D7']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Nimbus Roman','DejaVu Serif'],
 'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'axes.linewidth':0.6})
# filled = no delimiters, hollow = with delimiters; color = base configuration
base={'D0':'none','D1':'none','D2':'rule','D5':'rule','D3':'rem','D6':'rem','D7':'rr','D4':'rr'}
col={'none':'#000000','rule':'#56B4E9','rem':'#009E73','rr':'#CC79A7'}
dl={'D1','D4','D5','D6'}
lab={'D0':'D0','D1':'D1 Del','D2':'D2 Rule','D3':'D3 Rem','D4':'D4 Del+Rule+Rem','D5':'D5 Del+Rule','D6':'D6 Del+Rem','D7':'D7 Rule+Rem'}
off={'D0':(-4.8,2.0),'D1':(-8.5,1.0),'D2':(1.2,-2.6),'D3':(-9.3,2.6),'D4':(1.4,-2.9),'D5':(1.5,-2.9),'D6':(1.3,0.9),'D7':(-4.5,-3.6)}
fig,ax=plt.subplots(figsize=(3.45,2.45))
for c in C:
    m=a['main'][c]; x=m['asr']; y=m['task']; cc=col[base[c]]
    ax.errorbar(x['p']*100,y['p']*100,xerr=[[(x['p']-x['lo'])*100],[(x['hi']-x['p'])*100]],
        yerr=[[(y['p']-y['lo'])*100],[(y['hi']-y['p'])*100]],fmt='o',ms=4.5,color=cc,ecolor=cc,elinewidth=0.6,capsize=1.5,
        mfc='white' if c in dl else cc,mew=1.1,zorder=3)
    ax.annotate(lab[c],(x['p']*100+off[c][0],y['p']*100+off[c][1]),fontsize=6,color=cc)
ax.set_xlabel('Attack success rate (%)'); ax.set_ylabel('Task progress (%)')
ax.set_xlim(0,58); ax.set_ylim(70,94); ax.grid(alpha=0.25,lw=0.4)
ax.plot([],[],'o',color='0.3',ms=4,label='without delimiters'); ax.plot([],[],'o',color='0.3',mfc='white',ms=4,label='with delimiters')
ax.legend(fontsize=6,frameon=False,loc='lower right',handletextpad=0.2)
fig.tight_layout(pad=0.3); fig.savefig('analysis/fig_tradeoff.pdf'); fig.savefig('analysis/fig_tradeoff.png',dpi=300)

# delimiter effect: paired comparison per base configuration
pairs=[('No other defense','D0','D1'),('Rule','D2','D5'),('Reminder','D3','D6'),('Rule + reminder','D7','D4')]
fig,ax=plt.subplots(figsize=(3.45,1.75))
for j,(nm,a0,a1) in enumerate(pairs):
    p0=a['main'][a0]['asr']['p']*100; p1=a['main'][a1]['asr']['p']*100; cc=col[base[a0]]
    ax.annotate('',xy=(p1,j),xytext=(p0,j),arrowprops=dict(arrowstyle='->',color=cc,lw=1.1,shrinkA=3,shrinkB=3))
    ax.plot(p0,j,'o',color=cc,ms=4.5); ax.plot(p1,j,'o',color=cc,mfc='white',mew=1.1,ms=4.5)
    d=(a['main'][a1]['asr']['k']-a['main'][a0]['asr']['k'])/4
    ax.text(max(p0,p1)+1.5,j,f'{a0}\u2192{a1}  +{r1(d)} pp',va='center',fontsize=6.3,color=cc)
ax.set_yticks(range(4)); ax.set_yticklabels([p[0] for p in pairs]); ax.invert_yaxis()
ax.set_xlim(0,78); ax.set_xlabel('Attack success rate (%)'); ax.grid(axis='x',alpha=0.25,lw=0.4)
ax.set_ylim(3.6,-0.6)
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig('analysis/fig_delim.pdf'); fig.savefig('analysis/fig_delim.png',dpi=300)

cats=['unauthorized_action','data_modification','denial_of_service','exfiltration']
nm={'unauthorized_action':'Unauthorized action','data_modification':'Data modification','denial_of_service':'Denial of service','exfiltration':'Exfiltration'}
order=['D0','D1','D2','D5','D3','D6','D7','D4']
M=np.array([[a['cat'][k][c]['asr']*100 for c in order] for k in cats])
fig,ax=plt.subplots(figsize=(3.45,1.7))
im=ax.imshow(M,cmap='Greys',vmin=0,vmax=70,aspect='auto')
for i in range(len(cats)):
    for j in range(len(order)):
        ax.text(j,i,f'{M[i,j]:.0f}',ha='center',va='center',fontsize=6.5,color='white' if M[i,j]>40 else 'black')
ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
ax.set_yticks(range(len(cats))); ax.set_yticklabels([f"{nm[k]} (n={a['cat'][k]['D0']['n']})" for k in cats],fontsize=6.5)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
cb=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label('ASR (%)',fontsize=6.5); cb.outline.set_linewidth(0.4)
fig.tight_layout(pad=0.3); fig.savefig('analysis/fig_category.pdf'); fig.savefig('analysis/fig_category.png',dpi=300)

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, collections, itertools, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from score import score_episode, mcnemar_exact, wilson
rows={r['id']:r for r in map(json.loads,open('data/train.jsonl',encoding='utf-8'))}
CONDS=['D0','D1','D2','D3','D4','D5','D6','D7']
COMP={"D0":set(),"D1":{"D1"},"D2":{"D2"},"D3":{"D3"},"D4":{"D1","D2","D3"},"D5":{"D1","D2"},"D6":{"D1","D3"},"D7":{"D2","D3"}}
E={c:{e['id']:e for e in map(json.loads,open(f'results/{c}.jsonl',encoding='utf-8'))} for c in CONDS}
ids=sorted(E['D0'])
READ=('get_','search_','list_','check_','view_','read_','find_','lookup_')
recs=[]
for c in CONDS:
    for i in ids:
        e=E[c][i]; s=score_episode(e,rows[i]); r=rows[i]; t=r['injection']['injection_text'].strip()
        tmpl='priority' if t.startswith('[PRIORITY') else ('urgent' if 'URGENT' in t[:20] else 'other')
        w=[x['name'] for x in e['calls'] if not x['name'].startswith(READ)]
        recs.append(dict(cond=c,id=i,domain=r['domain'],cat=r['attack_category'],tmpl=tmpl,
            dl=int('D1' in COMP[c]),rule=int('D2' in COMP[c]),rem=int('D3' in COMP[c]),
            first=bool(s['attacked'] and w and w[0]==r['target_tool']),
            only_target=bool(s['attacked'] and all(n==r['target_tool'] for n in w)),
            ncalls=len(e['calls']),**s))
df=pd.DataFrame(recs); out={}
main={}
for c in CONDS:
    d=df[df.cond==c]
    def w(k,sub=d): k_=int(sub[k].sum()); p,lo,hi=wilson(k_,len(sub)); return dict(k=k_,n=len(sub),p=p,lo=lo,hi=hi)
    main[c]=dict(exposure=w('seen'),asr=w('attacked'),asr_strict=w('attacked_strict'),task=w('task_progress'),turn_limit=w('turn_limit'),
        med_s=float(d.seconds.median()),mean_tok=float(d.out_tok.mean()),ncalls=float(d.ncalls.mean()),
        first_among_att=float(d[d.attacked].first.mean()),n_first=int(d.first.sum()),only_target=int(d.only_target.sum()))
out['main']=main
P={c:df[df.cond==c].set_index('id') for c in CONDS}
pairs=[]
for a,b in itertools.combinations(CONDS,2):
    for k in ['attacked','task_progress']:
        x=int((P[a][k]&~P[b][k]).sum()); y=int((P[b][k]&~P[a][k]).sum())
        pairs.append(dict(a=a,b=b,metric=k,a_only=x,b_only=y,p=mcnemar_exact(x,y)))
for k in ['attacked','task_progress']:
    sub=sorted([p for p in pairs if p['metric']==k],key=lambda p:p['p']); m=len(sub); run=0
    for j,p in enumerate(sub): run=max(run,min(1,(m-j)*p['p'])); p['p_holm']=run
out['pairs']=pairs
rng=np.random.default_rng(7); B=10000; boot={}
A={c:{k:P[c].loc[ids,k].to_numpy().astype(float) for k in ['attacked','task_progress']} for c in CONDS}
idx=rng.integers(0,len(ids),(B,len(ids)))
for a,b in [(c,'D0') for c in CONDS[1:]]+[('D4','D3'),('D7','D3'),('D6','D3'),('D5','D2'),('D4','D7'),('D4','D6')]:
    for k in ['attacked','task_progress']:
        diff=A[a][k]-A[b][k]; bs=diff[idx].mean(1)
        boot[f'{a}-{b}:{k}']=dict(d=float(diff.mean()),lo=float(np.percentile(bs,2.5)),hi=float(np.percentile(bs,97.5)))
out['boot']=boot
df['y']=df.attacked.astype(int); dd=df[df.tmpl!='other']
def gee(formula):
    g=smf.gee(formula,groups='id',data=dd,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci=g.conf_int()
    return {k:dict(OR=float(np.exp(g.params[k])),lo=float(np.exp(ci.loc[k,0])),hi=float(np.exp(ci.loc[k,1])),p=float(g.pvalues[k])) for k in g.params.index}, float(g.cov_struct.dep_params)
fx='C(cat, Treatment("unauthorized_action")) + C(tmpl, Treatment("priority"))'
out['gee_cond'],out['gee_cond_corr']=gee('y ~ C(cond, Treatment("D0")) + '+fx)
out['gee_main'],_=gee('y ~ dl + rule + rem + '+fx)
out['gee_int'],_=gee('y ~ (dl + rule + rem)**2 + '+fx)
cnt=df.groupby('id').attacked.sum(); out['per_record']=dict(collections.Counter(int(v) for v in cnt))
for grp in ['cat','domain','tmpl']:
    out[grp]={gv:{c:dict(n=int((sub.cond==c).sum()),asr=float(sub[sub.cond==c].attacked.mean())) for c in CONDS} for gv,sub in df.groupby(grp)}
d0=set(P['D0'].index[P['D0'].attacked])
out['blocked']={c:int((~P[c].loc[sorted(d0),'attacked']).sum()) for c in CONDS[1:]}
out['hours']={c:float(df[df.cond==c].seconds.sum()/3600) for c in CONDS}
json.dump(out,open('analysis/analysis.json','w'),indent=1,default=float)
print({c:(round(main[c]['asr']['p']*100,1),round(main[c]['task']['p']*100,1),main[c]['med_s'],round(main[c]['mean_tok']),round(main[c]['ncalls'],2),round(main[c]['first_among_att'],3),main[c]['only_target']) for c in CONDS})
print('HOLM attacked:'); [print(' ',p['a'],p['b'],p['a_only'],p['b_only'],round(p['p'],4),round(p['p_holm'],4)) for p in pairs if p['metric']=='attacked']
print('HOLM task:'); [print(' ',p['a'],p['b'],p['a_only'],p['b_only'],round(p['p'],4),round(p['p_holm'],4)) for p in pairs if p['metric']=='task_progress' and p['p']<0.2]
for k,v in boot.items(): print(k, {a:round(b*100,1) for a,b in v.items()})
for name in ['gee_cond','gee_main','gee_int']: print(name,{k:(round(v['OR'],2),round(v['lo'],2),round(v['hi'],2),round(v['p'],4)) for k,v in out[name].items() if k!='Intercept'})
print('corr',out['gee_cond_corr']); print('per_record',out['per_record']); print('blocked',out['blocked']); print('hours',{k:round(v,2) for k,v in out['hours'].items()}, round(sum(out['hours'].values()),2))
print('cat',{k:{c:round(v[c]['asr']*100) for c in CONDS} for k,v in out['cat'].items()})
print('domain',{k:{c:round(v[c]['asr']*100) for c in CONDS} for k,v in out['domain'].items()})

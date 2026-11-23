import json
from decimal import Decimal, ROUND_HALF_UP
a=json.load(open('analysis/analysis.json'))
def r1(x): return str(Decimal(str(round(x,6))).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP))
def sg(x):
    s=r1(x); return ('+'+s if x>0 and s!='0.0' else ('$-$'+s[1:] if s.startswith('-') else s))
def fp(p):
    if p<0.001: return '$<$0.001'
    return f'{p:.3f}' if p<0.0995 or True else ''
comp={'D0':'','D1':'D','D2':'R','D3':'M','D4':'DRM','D5':'DR','D6':'DM','D7':'RM'}
names={'D0':'D0 none','D1':'D1 Del','D2':'D2 Rule','D3':'D3 Rem','D5':'D5 Del+Rule','D6':'D6 Del+Rem','D7':'D7 Rule+Rem','D4':'D4 Del+Rule+Rem'}
order=['D0','D1','D2','D3','D5','D6','D7','D4']
ph={}
for p in a['pairs']:
    if p['a']=='D0': ph[(p['b'],p['metric'])]=p['p_holm']
rows=[]
for c in order:
    m=a['main'][c]; A=m['asr']; T=m['task']
    asr=f"{r1(A['p']*100)} [{r1(A['lo']*100)}, {r1(A['hi']*100)}]"
    task=f"{r1(T['p']*100)} [{r1(T['lo']*100)}, {r1(T['hi']*100)}]"
    if c=='D0': da=dt=pa=pt='--'
    else:
        b=a['boot'][f'{c}-D0:attacked']; bt=a['boot'][f'{c}-D0:task_progress']
        da=f"{sg(b['d']*100)} [{sg(b['lo']*100)}, {sg(b['hi']*100)}]"
        dt=f"{sg(bt['d']*100)} [{sg(bt['lo']*100)}, {sg(bt['hi']*100)}]"
        pa=fp(ph[(c,'attacked')]); pt=fp(ph[(c,'task_progress')])
    bold=c=='D7'
    if bold: asr=asr.replace(r1(A['p']*100),'\\textbf{'+r1(A['p']*100)+'}',1)
    strict=r1(m['asr_strict']['p']*100)
    rows.append(f"{names[c]} & {asr} & {strict} & {da} & {pa} & {task} & {dt} & {pt} \\\\")
open('analysis/table2_rows.tex','w').write('\n'.join(rows)+'\n')
print('\n'.join(rows))
# numbers for text
for c in order: print(c,'exp',r1(a['main'][c]['exposure']['p']*100),'gap',r1((a['main'][c]['asr']['p']-a['main'][c]['asr_strict']['p'])*100),'turnlim',r1(a['main'][c]['turn_limit']['p']*100),'med',a['main'][c]['med_s'],'tok',round(a['main'][c]['mean_tok']),'calls',round(a['main'][c]['ncalls'],2),'k',a['main'][c]['asr']['k'])
for k in ['D4-D3','D7-D3','D6-D3','D5-D2','D4-D7','D4-D6']:
    b=a['boot'][k+':attacked']; t=a['boot'][k+':task_progress']; print(k,sg(b['d']*100),sg(b['lo']*100),sg(b['hi']*100),'| task',sg(t['d']*100),sg(t['lo']*100),sg(t['hi']*100))

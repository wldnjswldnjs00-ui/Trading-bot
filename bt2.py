"""v2: risk 1% of equity per trade, stop variants, in/out-of-sample split."""
import sys, numpy as np, pandas as pd, itertools
from engine import atr, zigzag, internal_count, fib_score
df = pd.read_csv('data/btcusdt_1h.csv'); df['time']=pd.to_datetime(df.time,format='ISO8601',utc=True)
A = atr(df); o,h,l,c = df.o.values,df.h.values,df.l.values,df.c.values
FEE=0.0012; RISK=0.01; SPLIT=pd.Timestamp('2023-01-01',tz='UTC')
def run(K, stopmode, TGT, rlo, rhi, sub):
    big=zigzag(df,K,A); low=zigzag(df,K/2.5,A); out=[]; busy=-1
    for j in range(2,len(big)):
        p0,p1,p2=big[j-2],big[j-1],big[j]; up=p2[2]==-1; s=1 if up else -1
        w1=s*(p1[1]-p0[1]); r=s*(p1[1]-p2[1])/w1 if w1>0 else 9
        if not (rlo<=r<rhi): continue
        known=p2[3]
        if sub:
            lk=[p for p in low if p[3]<=known]
            if not internal_count(lk,p0[0],p1[0],up,5) or not internal_count(lk,p1[0],p2[0],up,3): continue
        e=known+1
        if e>=len(o) or e<=busy: continue
        entry=o[e]; stop=p0[1] if stopmode=='w1' else p2[1]-s*0.5*A[known]
        tgt=p2[1]+s*TGT*w1
        risk=s*(entry-stop)/entry
        if risk<=0.002 or s*(tgt-entry)<=0: continue
        for t in range(e,len(o)):
            if (l[t]<=stop) if up else (h[t]>=stop): x=stop;break
            if (h[t]>=tgt) if up else (l[t]<=tgt): x=tgt;break
        else: x=c[-1]
        R=(s*(x-entry)/entry-FEE)/risk
        out.append((df.time[e],R)); busy=t
    return out
def stats(tr):
    if len(tr)<5: return None
    R=np.array([x[1] for x in tr]); eq=np.cumprod(1+RISK*R)
    return len(R),(R>0).mean(),R.mean(),eq[-1]-1,(eq/np.maximum.accumulate(eq)-1).min()
rows=[]
for K,sm,T,(rlo,rhi),sub in itertools.product([4,6,8,12],['w1','w2'],[1.0,1.618,2.618],[(0.236,1),(0.382,0.786),(0.5,0.886)],[True,False]):
    tr=run(K,sm,T,rlo,rhi,sub)
    a=stats([x for x in tr if x[0]<SPLIT]); b=stats([x for x in tr if x[0]>=SPLIT])
    if a and b: rows.append((K,sm,T,rlo,rhi,sub,*a,*b))
R=pd.DataFrame(rows,columns='K stop tgt rlo rhi sub n_in win_in avgR_in tot_in mdd_in n_out win_out avgR_out tot_out mdd_out'.split())
R.to_csv('grid.csv',index=False)
print('configs',len(R),'| profitable in-sample',(R.avgR_in>0).sum(),'| both',( (R.avgR_in>0)&(R.avgR_out>0)).sum())
best=R.sort_values('avgR_in',ascending=False).head(8)
print(best.round(3).to_string(index=False))

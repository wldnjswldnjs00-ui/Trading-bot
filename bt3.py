"""v3: stop = previous swing low/high (lower degree), max stop distance 1-2%.
Big degree gives wave 1; inside the wave-2 retrace zone, the first lower-degree
swing low that confirms = entry. Stop just beyond that swing. Causal."""
import numpy as np, pandas as pd, itertools
from engine import atr, zigzag
df=pd.read_csv('data/btcusdt_1h.csv'); df['time']=pd.to_datetime(df.time,format='ISO8601',utc=True)
A=atr(df); o,h,l,c=df.o.values,df.h.values,df.l.values,df.c.values
FEE=0.0012; RISK=0.01; SPLIT=pd.Timestamp('2023-01-01',tz='UTC'); BUF=0.001
def run(K,ratio,maxstop,TGT,zone):
    big=zigzag(df,K,A); low=zigzag(df,K/ratio,A)
    lk=np.array([p[3] for p in low]); out=[]; busy=-1
    for j in range(1,len(big)):
        p0,p1=big[j-1],big[j]; up=p1[2]==1; s=1 if up else -1
        w1=s*(p1[1]-p0[1]);
        if w1<=0: continue
        nxt=big[j+1][3] if j+1<len(big) else len(o)   # until next big pivot is known
        # lower-degree counter-swings (wave-2 ends) confirmed after p1 known
        for q in low:
            if q[3]<=max(p1[3],busy) or q[3]>=nxt: continue
            if q[2]!=(-1 if up else 1) or q[0]<=p1[0]: continue
            r=s*(p1[1]-q[1])/w1
            if not (zone[0]<=r<zone[1]): continue
            if s*(q[1]-p0[1])<=0: break                       # wave 2 beyond start of 1: count dead
            e=q[3]+1
            if e>=len(o): break
            entry=o[e]; stop=q[1]*(1-s*BUF); risk=s*(entry-stop)/entry
            if risk<=0 or risk>maxstop: continue
            tgt=q[1]+s*TGT*w1
            if s*(tgt-entry)<=0: continue
            for t in range(e,len(o)):
                if (l[t]<=stop) if up else (h[t]>=stop): x=stop;break
                if (h[t]>=tgt) if up else (l[t]<=tgt): x=tgt;break
            else: x=c[-1]
            out.append((df.time[e],(s*(x-entry)/entry-FEE)/risk,risk)); busy=t
            break
    return out
def st(tr):
    if len(tr)<5: return (len(tr),0,0,0,0)
    R=np.array([x[1] for x in tr]); eq=np.cumprod(1+RISK*R)
    return len(R),(R>0).mean(),R.mean(),eq[-1]-1,(eq/np.maximum.accumulate(eq)-1).min()
rows=[]
for K,ra,ms,T,z in itertools.product([4,6,8,12],[3,5],[0.01,0.015,0.02],[1.0,1.618,2.618],[(0.382,0.886),(0.5,0.786)]):
    tr=run(K,ra,ms,T,z)
    rows.append((K,ra,ms,T,z[0],*st([x for x in tr if x[0]<SPLIT]),*st([x for x in tr if x[0]>=SPLIT]),np.mean([x[2] for x in tr]) if tr else 0))
R=pd.DataFrame(rows,columns='K ratio maxstop tgt zlo n_in win_in avgR_in tot_in mdd_in n_out win_out avgR_out tot_out mdd_out avgstop'.split())
R.to_csv('grid3.csv',index=False)
ok=R[(R.n_in>=20)&(R.n_out>=15)].copy(); ok['m']=ok[['avgR_in','avgR_out']].min(axis=1)
print('configs',len(R),'enough trades',len(ok),'positive both',(ok.m>0).sum(),'median out',ok.avgR_out.median().round(3))
print(ok.sort_values('m',ascending=False).head(10).round(3).to_string(index=False))

import pickle,itertools,numpy as np
from pathlib import Path
s=pickle.loads(Path('/holdout_task/public_batch.pkl').read_bytes())[0]['study']
def fit(X,Y): return np.linalg.lstsq(X,Y,rcond=None)[0]
def aff(u): return np.column_stack((np.ones(len(u)),u))
def robust(X,Y):
    W=np.ones(len(X))
    for _ in range(25):
        B=fit(X*W[:,None]**.5,Y*W[:,None]**.5)
        r=np.linalg.norm(Y-X@B,axis=1)
        W=np.minimum(1,1.45*np.median(r)/np.maximum(1e-12,r))
    return B,r

def blocks(u):
    x=u.T; pairs=list(itertools.combinations(range(4),2)); diff=np.array([x[i]-x[j] for i,j in pairs]); sums=np.array([x[i]+x[j] for i,j in pairs]); cross=np.array([x[i]*x[j] for i,j in pairs]);
    out={'sq':x*x,'cross':cross,'pairdot':(x[0]*x[2]+x[1]*x[3])[None,:],'pairadj':(x[0]*x[1]+x[2]*x[3])[None,:],'cube':x**3,'trip':np.array([x[i]*x[j]*x[k] for i,j,k in itertools.combinations(range(4),3)])}
    for w in [0.5,1,1.2,1.5,1.7,2,2.5,3,np.pi]:
        for name,a in [('x',x),('d',diff),('s',sums),('p',cross)]:
            for fun in [np.sin,np.cos]: out[f'{fun.__name__}_{name}_{w:.3g}']=fun(w*a)
    return {n:a.T for n,a in out.items()}

def score(X,Y):
    Q,R=np.linalg.qr(X,mode='reduced'); h=np.sum(Q*Q,axis=1); r=Y-Q@(Q.T@Y); return np.mean(np.abs(r/np.maximum(1e-10,1-h)[:,None])),np.mean(np.abs(r)),np.linalg.cond(R)

if __name__=='__main__':
 for panel in ['panel_a','panel_b']:
    u=s[panel]['u']; v=s[panel]['v']; x=aff(u); b,r=robust(x,v); print(panel,'affine',b.round(4),'row residuals',np.sort(r).round(4)); print('affine CV,train,cond',score(x,v));
    if panel=='panel_a': continue
    d=blocks(u); print('best blocks', sorted((score(np.column_stack((x,f)),v),n) for n,f in d.items())[:18]);
    rank=[]
    for nn in itertools.combinations(d,2):
      X=np.column_stack((x,*(d[n] for n in nn))); a=score(X,v); rank.append((a,nn))
    print('best double blocks',sorted(rank)[:25]);
    cols={f'{n}:{j}':f[:,j] for n,f in d.items() for j in range(f.shape[1])};
    for j in range(4):
      print('channel',j,'best singles',sorted((score(np.column_stack((x,f)),v[:,j:j+1]),n) for n,f in cols.items())[:9]);
      beam=[(score(x,v[:,j:j+1]),())]; seen=set()
      for k in range(4):
        nxt=[]
        for _,ns in beam:
          for n in cols:
            ts=tuple(sorted((*ns,n)))
            if n in ns or ts in seen: continue
            seen.add(ts); X=np.column_stack((x,*(cols[t] for t in ts))); a=score(X,v[:,j:j+1]);
            if a[2]<1e6: nxt.append((a,ts))
        beam=sorted(nxt)[:15]; print('beam',k+1,beam[:4]);

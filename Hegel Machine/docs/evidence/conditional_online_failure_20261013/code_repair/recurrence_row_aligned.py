# Host repair of the supervised row contract; scientific model family unchanged.
import json
import numpy as np
from scipy.optimize import least_squares

rows=json.load(open('observations.json'))
seqs=[]
for r in rows:
    h=r['history']+[r['features']]
    u=np.array([[float(v.get(k,0.)) for k in ('x','z','c')] for v in h], dtype=float)
    seqs.append((u,float(r['target'])))

def predict_seq(u,p):
    a=p[0]; b=p[1:4]; d=p[4:7]; q=p[7:]
    m=0.; prev=np.zeros(3); out=[]
    for v in u:
        m0=m
        m=a*m0+np.dot(b,v)+np.dot(d,prev)
        f=np.array([1.,m,m*m,m0,m0*m0,m*m0,*v,m*v[0],m*v[1],m*v[2]])
        out.append(np.dot(q,f))
        prev=v
    return np.asarray(out)

def residual(p):
    return np.asarray([predict_seq(u,p)[-1]-y for u,y in seqs])

best=None
for seed in range(5):
    rng=np.random.default_rng(seed)
    p0=np.r_[rng.uniform(.1,.9),rng.normal(0,.2,6),rng.normal(0,.2,12)]
    opt=least_squares(residual,p0,max_nfev=1200)
    e=residual(opt.x)
    result=(float(np.mean(e*e)),float(np.max(np.abs(e))),opt.x)
    if best is None or result[0]<best[0]: best=result
print('rows',len(rows),'best_mse',best[0],'max_abs',best[1])
print('params',best[2].tolist())
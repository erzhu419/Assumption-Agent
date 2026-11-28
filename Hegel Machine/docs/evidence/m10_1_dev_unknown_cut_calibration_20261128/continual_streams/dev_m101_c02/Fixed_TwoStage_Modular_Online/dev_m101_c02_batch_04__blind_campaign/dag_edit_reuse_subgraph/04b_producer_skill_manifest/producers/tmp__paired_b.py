from panels import p, np
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
import sys
sys.path.insert(0, '/holdout_task')
from node_00 import _basis_b
u,v=(p['panel_b'][k] for k in ('u','v'))
pairs=((0,3),(1,2))
def design(x,w,j,kind):
    a,b=pairs[j]; s,t=x[:,a],x[:,b]; H=np.c_[np.ones(len(x)),s,t,np.sin(w*s),np.sin(w*t)]
    if kind=='full harmonic': return _basis_b(x,w)
    if kind=='paired + cross': return np.c_[H,s*t]
    if kind=='paired + quadratic': return np.c_[H,s*t,s*s,t*t]
    if kind=='paired + second harmonic': return np.c_[H,np.sin(2*w*s),np.sin(2*w*t),np.cos(2*w*s),np.cos(2*w*t)]
    return H
def fit(idx,kind):
    def profile(w, predict=None):
        out=[]
        for j in range(2):
            X=design(u[idx],w,j,kind); y=v[idx,2*j:2*j+2]; c=np.linalg.lstsq(X,y,rcond=None)[0]
            out.append(X@c-y if predict is None else design(predict,w,j,kind)@c)
        return np.concatenate(out,axis=1)
    f=lambda w: np.sum(profile(w)**2)
    w=min(np.linspace(.6,5,45),key=f); w=minimize_scalar(f,bounds=(w-.1,w+.1),method='bounded').x
    return w,profile
for kind in ('full harmonic','paired','paired + cross','paired + quadratic','paired + second harmonic'):
    errors=[]
    for k in range(len(u)):
        idx=np.delete(np.arange(len(u)),k); w,F=fit(idx,kind); errors.append(F(w,u[k:k+1])[0]-v[k])
    w,F=fit(np.arange(len(u)),kind); e=np.linalg.norm(errors,axis=1)
    print(kind,'frequency',round(w,5),'fit/LOO/worst LOO row',*(round(z,6) for z in (np.sqrt(np.mean(np.sum(F(w)**2,axis=1))),np.sqrt(np.mean(e**2)),max(e))))

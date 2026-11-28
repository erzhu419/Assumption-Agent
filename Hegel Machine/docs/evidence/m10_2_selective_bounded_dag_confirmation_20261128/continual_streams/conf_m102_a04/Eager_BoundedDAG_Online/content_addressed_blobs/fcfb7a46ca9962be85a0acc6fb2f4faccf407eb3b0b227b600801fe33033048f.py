import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve
from itertools import combinations_with_replacement

def ls(a,b): return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def one(a): return np.c_[np.ones(len(a)),a]
def pair(a): return a*np.roll(a,-1,axis=1)
def align(panel,x):
    u,v=panel['u'],panel['v']
    def f(a): return one(np.c_[a,pair(a),a**3])
    F=f(u); mask=np.max(abs(v-np.median(v,axis=0)),axis=1)<6
    for _ in range(3):
        c=ls(F[mask],v[mask]); r=np.linalg.norm(F@c-v,axis=1); mask=r<max(7*np.median(r),.001)
    c[abs(c)<.001]=0
    return np.array([least_squares(lambda a:(f(a[None])@c)[0]-q,np.linalg.solve(c[1:5].T,q-c[0]),gtol=1e-11).x for q in x])
def features(s,z):
    b=s['panel_b']; w=one(z)@ls(one(b['u']),b['v']); p=s['panel_c']; u=p['u']; cp=ls(one(pair(u)),p['v1']); cq=ls(one(np.cos(2.4*u)),p['v2']); c=one(pair(z))@cp; d=one(np.cos(2.4*z))@cq
    return w,c,d

def solve_batch(study):
    tr=study['train']; n=len(tr['y']); z=align(study['panel_a'],np.vstack([tr['x'],study['test']['x']])); w,c,d=features(study,z)
    X=np.c_[z,pair(z),np.cos(2.4*z)]
    X=X/np.maximum(X[:n].std(0),.1); G=X@X.T/X.shape[1]; D=np.maximum(G.diagonal()[:,None]+G.diagonal()-2*G,0)
    K=(1+.45*G)**3+.1*np.exp(-.25*D)
    Y=np.c_[tr['y'],tr['traj']]; pred=K[n:,:n]@solve(K[:n,:n]+.002*np.eye(n),Y,assume_a='pos'); y,T=pred[:,0],pred[:,1:]
    return dict(y_hat=y,state_pred=T,diagnostics_vec=np.array([y.mean(),y.std(),T[:,-1].mean(),T[:,-1].std()]))

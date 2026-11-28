import numpy as np
from itertools import combinations

def _features(z):
    s = [np.ones(len(z)), z]
    for j in range(4):
        u = z[:,j:j+1]
        s.extend([u**2,u**3,u**4,np.tanh(u),np.arctan(u),u/(1+abs(u)),np.exp(-u*u/2)])
    groups = []
    for i,j in combinations(range(4),2):
        u,v=z[:,i:i+1],z[:,j:j+1]
        g=[u*v,u*u*v,u*v*v,u*u*v*v,u**3*v,u*v**3,u*v/(1+.5*(u*u+v*v))]
        for sign in (-1,1):
            w=(u+sign*v)/1.4
            g.extend([np.sin(w),np.cos(w),np.tanh(w),w/(1+abs(w)),np.exp(-w*w/2)])
        groups.append(np.column_stack(g))
    return s,groups

def solve_batch(study):
    tr=study['train']; x=np.asarray(tr['x']); xt=np.asarray(study['test']['x']); n=len(x)
    z=(np.vstack([x,xt])-x.mean(0))/np.maximum(x.std(0),1e-12)
    q=-np.expm1(-np.asarray(study['eval_times'])[:,None]*np.array([.392,1.857]))
    a=np.linalg.lstsq(q,tr['traj'].T,rcond=None)[0].T
    s,g=_features(z)
    f=np.column_stack(s+g+[np.prod(z[:,j],axis=1) for j in combinations(range(4),3)])[:,1:]
    ys=(2,5,10,14,16,23,24,26,33,35,40,50,51,52,54,61,63,64,65,67,77,78,85,87,92,93,96,101,102,104,106,111,119,128,132,137)
    fs=(7,9,10,24,30,31,35,36,39,40,51,54,61,68,71,72,85,86,96,101,104,106,111,112,123,127,128,134)
    def ols(ids,y):
        mat=np.column_stack([np.ones(len(z)),f[:,ids]])
        return mat[n:]@np.linalg.lstsq(mat[:n],y,rcond=None)[0]
    m=np.column_stack(s+[g[j] for j in (0,1,3)])
    m=m/np.maximum(np.linalg.norm(m[:n],axis=0),1e-8)
    slow=m[n:]@m[:n].T@np.linalg.solve(m[:n]@m[:n].T+.02*np.eye(n),a[:,0])
    y=ols(ys,tr['y']); pred=np.column_stack([slow,ols(fs,a[:,1])])@q.T
    return {'y_hat':y,'state_pred':pred,'diagnostics_vec':np.array([y.mean(),y.std(),pred[:,-1].mean(),pred[:,-1].std()])}

_sparse = solve_batch

def _panel_pred(study, blend, reg, kind):
    tr=study['train']; n=len(tr['x']); xx=np.vstack([tr['x'],study['test']['x']]);
    p=study['panel_a']; u=np.asarray(p['u']); v=np.asarray(p['v']);
    h=np.column_stack([np.ones(len(u)),u]); av=np.linalg.lstsq(h,v,rcond=None)[0]; r=np.linalg.norm(v-h@av,axis=1); keep=r<3
    gains=[]; offsets=[]
    for j in range(4):
        c=np.linalg.lstsq(h[keep][:,[0,j+1]],v[keep,j],rcond=None)[0]; offsets.append(c[0]);gains.append(c[1]);
    z=(xx-offsets)/gains
    if kind=='sat':
        p=study['panel_c']; u=np.asarray(p['u']); v=np.asarray(p['v2']); h=np.column_stack([np.ones(len(u)),u]); b=np.linalg.lstsq(h,v,rcond=None)[0];
        pz=np.column_stack([np.ones(len(u)),u,np.expm1(-.2962*u*u)]); w=np.linalg.lstsq(pz,v,rcond=None)[0];
        v=np.column_stack([np.ones(len(z)),z,np.expm1(-.2962*z*z)])@w
        z=z+blend*((v-np.column_stack([np.ones(len(z)),z])@b)@np.linalg.inv(b[1:]))
        t=np.tanh(z)
    else:
        z=z+blend*np.expm1(-.2962*z*z); t=None
    def quad(a):
        return [a]+[a[:,i]*a[:,j] for i in range(4) for j in range(i,4)]
    f=np.column_stack(quad(z)+(quad(t) if t is not None else [np.sin(z),np.cos(z),np.sin(2*z),np.cos(2*z)]))
    f=(f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-7); y=np.column_stack([tr['y'],tr['traj']]); ym=y.mean(0);
    return f[n:]@f[:n].T@np.linalg.solve(f[:n]@f[:n].T+reg*np.eye(n),y-ym)+ym

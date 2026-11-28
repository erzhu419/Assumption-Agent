import numpy as np
from scipy.optimize import least_squares

def fit(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def aff(x): return np.c_[np.ones(len(x)),x]

def maps(s):
    u,v=(s['panel_a'][k] for k in ('u','v')); ok=np.linalg.norm(v,axis=1)<5; u,v=u[ok],v[ok]
    w=np.roll(u,-1,axis=1); off,scale=np.array([fit(aff(u[:,j]),v[:,j]-.32*(u[:,j]-1)*w[:,j]-.16*u[:,j]**3) for j in range(4)]).T
    def f(z): return off+scale*z+.32*(z-1)*np.roll(z,-1,axis=-1)+.16*z**3
    def inv(x): return np.array([least_squares(lambda z:f(z)-h,(h-off)/scale,gtol=1e-12).x for h in x])
    def op(panel,k,fn,p,lo,hi):
        u,v=panel['u'],panel[k]
        def res(p):
            h=aff(fn(u,p));return (h@fit(h,v)-v).ravel()
        p=least_squares(res,p,bounds=(lo,hi),gtol=1e-12).x
        c=fit(aff(fn(u,p)),v)
        return lambda x: aff(fn(x,p))@c
    b=op(s['panel_b'],'v',lambda x,p:np.cos(p[0]*x),[2.6],[2],[3.2])
    c=op(s['panel_c'],'v1',lambda x,p:x/(1+p[0]*abs(x)**p[1])**p[2],[.62,1.07,1.57],[.1,.1,.2],[4,4,5])
    d=op(s['panel_c'],'v2',lambda x,p:np.logaddexp(0,p[0]*x+p[1]),[2.2,0],[1,-2],[4,2])
    return f,inv,b,c,d

def pool(s,x,depth=2):
    f,inv,b,c,d=maps(s); z=inv(x); D={'x':x,'z':z,'a':f(x)}; new=D.copy()
    for _ in range(depth):
        new={k+n:fn(X) for n,X in new.items() for k,fn in zip('bcd',(b,c,d))};D.update(new)
    return D

if __name__=='__main__':
    import pickle,itertools
    s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; T=t['traj']; D=pool(s,t['x'],2)
    print('range x,z',D['x'].min(0),D['x'].max(0),D['z'].min(0),D['z'].max(0)); print('traj svd',np.linalg.svd(T,compute_uv=False));print('y,T',np.c_[y,T][:6]); print('summary',t['summary'])
    def err(A,Y):
        h=aff(A); e=h[52:]@fit(h[:52],Y[:52])-Y[52:];return abs(e).mean()
    Q={}
    for k,x in D.items():
        Q[k]=np.stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)],1)
    for Y in [T,y]:
        print('output',Y.shape)
        for k in (1,2,3):
            if k==3: it=itertools.combinations([j for j in D if len(j)<=2],k)
            else: it=itertools.combinations(D,k)
            R=sorted((err(np.column_stack([D[j] for j in js]),Y),js) for js in it);print('affine',k,R[:10])
        R=sorted((err(np.c_[x,Q[k]],Y),k) for k,x in D.items());print('quadratic',R[:10]);
        R=[]
        for (j,a),(k,b) in itertools.combinations(D.items(),2):
            for label,h in [('prod',a*b),('quotA',a/(1+.22*b*b)),('quotB',b/(1+.22*a*a)),('roll',a*np.roll(b,-1,axis=1)),('diff',abs(a-b)),('quotS',a/(1+.22*(a-b)**2))]:
                R.append((err(np.c_[a,b,h],Y),j,k,label))
        print('mix',sorted(R)[:20])

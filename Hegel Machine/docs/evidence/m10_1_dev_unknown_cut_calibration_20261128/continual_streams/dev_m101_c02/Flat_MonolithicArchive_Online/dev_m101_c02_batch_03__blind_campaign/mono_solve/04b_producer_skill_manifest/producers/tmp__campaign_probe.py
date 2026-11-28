import pickle
import numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement, product

S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(a): return np.column_stack((np.ones(len(a)),a))
def poly(a,d): return np.column_stack([np.prod(a[:,idx],axis=1) for n in range(d+1) for idx in combinations_with_replacement(range(a.shape[1]),n)])
def calibrate(s, x):
    p=s['panel_a']; u,v=p['u'],p['v']; keep=np.max(abs(v-u),axis=1)<3; u,v=u[keep],v[keep]
    r=np.roll(u,-1,axis=1); n=len(u)
    F=np.column_stack((np.tile(np.eye(4),(n,1)),(u[:,:,None]*np.eye(4)).reshape(-1,4),u.ravel()**3,((u-1)*r).ravel()))
    c=ls(F,v.ravel()); b,g=c[:4],c[4:8]; a,h=c[8:];
    x=np.asarray(x); w=(x-b)/g
    for _ in range(15):
        residual=b+g*w+a*w**3+h*(w-1)*np.roll(w,-1,axis=1)-x
        if np.max(abs(residual))<1e-12: break
        J=(g+3*a*w*w+h*np.roll(w,-1,axis=1))[:,:,None]*np.eye(4)+h*(w-1)[:,:,None]*np.roll(np.eye(4),1,axis=1)
        w-=np.clip(np.linalg.solve(J,residual[...,None])[...,0],-.5,.5)
    return w,c,np.sqrt(np.mean((F@c-v.ravel())**2)),np.max(abs(residual))

if __name__=='__main__':
    np.set_printoptions(precision=5,suppress=True)
    tr=S['train']; x=tr['x']; T=tr['traj']; y=tr['y']; t=S['eval_times']; p=S['panel_b']; C=S['panel_c']; w,c,pe,ie=calibrate(S,np.r_[x,S['test']['x']]); print('A parameters, panel RMSE, inverse error, input ranges',c,pe,ie,w.min(0),w.max(0)); w=w[:len(x)]
    op=ls(aff(p['u']),p['v']); z=aff(w)@op; z0=z-op[0]
    c1=aff(w)@ls(aff(C['u']),C['v1']); c2=np.c_[aff(w),w**2]@ls(np.c_[aff(C['u']),C['u']**2],C['v2']); print('C2 additive square rmse',np.std(np.c_[aff(C['u']),C['u']**2]@ls(np.c_[aff(C['u']),C['u']**2],C['v2'])-C['v2']))
    rng=np.random.default_rng(14); idx=np.arange(len(x)); folds=[(idx[:56],idx[56:])]; folds += [(np.setdiff1d(idx,i),i) for i in np.array_split(rng.permutation(idx),5)]; nrm=np.linalg.norm(w,axis=1); order=np.argsort(nrm); folds.append((order[:-12],order[-12:])); target=np.c_[y,T]
    def evaluate(f):
        errs=np.array([np.mean(abs(f[j]@ls(f[i],target[i])-target[j]),axis=0) for i,j in folds]); return np.round(np.c_[errs[:,0],errs[:,1:].mean(1),errs[:,-1]][[0,-1]],5),np.round(np.c_[errs[:,0],errs[:,1:].mean(1),errs[:,-1]][1:-1].max(0),5)
    for name,u in [('raw',x),('cal',w),('B',z),('Bcenter',z0),('C1',c1),('C2',c2)]:
        print(name)
        for k,f in [('linear',aff(u)),('pairSum',np.c_[aff(u),u[:,0]*u[:,2]+u[:,1]*u[:,3]]),('pair',np.c_[aff(u),u[:,0]*u[:,2],u[:,1]*u[:,3]]),('sq',np.c_[aff(u),u**2]),('cubic',np.c_[aff(u),u**3]),('poly2',poly(u,2)),('poly3',poly(u,3))]:
            if name in ('C1','C2') and k not in ('linear','poly2'):continue
            e,m=evaluate(f); print(k,e[0],m,e[1])
    print('B scalar polynomial coefficients',ls(poly(z,3),y)); print('B first and final time quadratic coefficients',ls(poly(z,2),T[:,[0,-1]]).T); print('C2 scalar and trajectory linear coefficients',ls(aff(c2),target[:,[0,1,-1]]).T)
    _,sv,vt=np.linalg.svd(T,full_matrices=False); funcs={'exp':lambda r:-np.expm1(-t*r),'log':lambda r:np.log1p(t*r),'hyp':lambda r:t/(1+t*r),'sin':lambda r:np.sin(t*r),'lin':lambda r:t,'tanh':lambda r:np.tanh(t*r)}; weights=sv[:2,None]*vt[:2]; out=[]
    for k,l in combinations_with_replacement(funcs,2):
        def design(r):return np.column_stack((funcs[k](r[0]),funcs[l](r[1])))
        def residual(r):
            d=design(r);return (d@ls(d,weights.T)-weights.T).ravel()
        res=min([least_squares(residual,r,bounds=(.001,8),max_nfev=100) for r in ([.6,1.7],[1.7,.6],[.2,3.5])],key=lambda a:a.cost);out.append((np.linalg.norm(res.fun),k,l,res.x))
    print('temporal families',*sorted(out,key=lambda a:a[0])[:12],sep='\n')

import sys,pickle,numpy as np
from itertools import combinations_with_replacement
sys.path.insert(0,'/holdout_task')
from batch_solver import _panels,_aff,_quad,_ols,solve_batch

def _library(study,x,y,tr):
    z=_panels(study,x); n=len(y); q=_quad(x)
    roots={k:z[k] for k in ('x','a','i','b','j','ib','ij','bi','ji')}
    cov=np.zeros((4,4)); cov[np.triu_indices(4)]=_ols(q[:n],y)[5:];cov=(cov+cov.T)/2
    _,v=np.linalg.eigh(cov);roots['e']=x@v[:,[0,3,1,2]]
    _,_,v=np.linalg.svd(tr,full_matrices=False)
    amp=tr@v[:2].T
    h=q@_ols(q[:n],amp)
    yield ('state','xamp'),np.c_[q,h*h,h[:,0]*h[:,1],(x[:,:,None]*h[:,None,:]).reshape(-1,8)]
    for key,d in roots.items():
        if key=='e':
            yield (key,'ridges'),np.c_[q,d[:,:2]**3,d[:,:2]**4,d[:,:2]**5]
        cubic=np.column_stack([np.prod(d[:,idx],axis=1) for idx in combinations_with_replacement(range(4),3)])
        yield (key,'cube'),np.c_[q,cubic]
        yield (key,'diag'),np.c_[q,d**3,d**4,d**5]
        pairs=np.column_stack([d[:,i]+sign*d[:,j] for i in range(4) for j in range(i+1,4) for sign in (-1,1)])
        for p in (0.8,1.45,2.15):
            for mode in ('s','t','r'):
                if mode=='s': v=np.sin(p*d); w=np.cos(p*d)
                elif mode=='t': v=np.tanh(p*d);w=v*v
                else: v=d/(1+(p*d)**2);w=v*v
                yield (key,mode,p,'sat'),_quad(v)
                yield (key,mode,p,'mix'),np.c_[q,v,w,(d[:,:,None]*v[:,None,:]).reshape(-1,16)]
                yield (key,mode,p,'add'),np.c_[q,v,w]
            yield (key,'pairs',p),np.c_[q,np.sin(p*pairs),np.cos(p*pairs)]
        if key=='e':continue
        v,w=z[key+'1'],z[key+'2']
        for label,a in [('1',v),('2',w),('12',np.c_[v,w])]:
            yield (key,label,'add'),np.c_[q,a,a*a]
            yield (key,label,'xprod'),np.c_[q,a,a*a,(x[:,:,None]*a[:,None,:]).reshape(len(x),-1)]
            yield (key,label,'quad'),_quad(np.c_[x,a])
        for label in ('1','1j','2','2j'):
            a=z[key+label]
            for p in (1.,1.7,2.3):
                for mode in ('s','r'):
                    v=np.sin(p*a)/p if mode=='s' else a/(1+(p*a)**2)
                    yield (key,label,p,mode),np.c_[q,v,(a[:,:,None]*v[:,None,:]).reshape(-1,16)]

def _fits(f,y):
    n=len(y); scale=np.maximum(f[:n].std(0),.05);scale[0]=1
    f=f/scale;U,S,V=np.linalg.svd(f[:n],full_matrices=False); sq=S*S; J=f[n:]@V.T
    for link in (0,1,1.4,1.75,2.4,-1.,-1.6):
        if link>0 and abs(y).max()>=link:continue
        t=y if not link else np.arctanh(y/link)*link if link>0 else np.sinh(y/(-link))*(-link)
        u=U.T@t
        def inv(x):return x if not link else link*np.tanh(x/link) if link>0 else -link*np.arcsinh(x/(-link))
        for l in (.006,.03,.15,.55,1.7):
            fac=sq/(sq+l); loo=t-(t-U@(fac*u))/np.maximum(1-U**2@fac,1e-8)
            yield (link,l),inv(J@(S*u/(sq+l))),np.mean(abs(inv(loo)-y))

def _predict(study, x, y, tr, selected):
    s=dict(selected);pred=[]
    for k,f in _library(study,x,y,tr):
        if k in s:
            g={m:p for m,p,_ in _fits(f,y)}
            pred.extend(g[m] for m in s[k] if m in g)
    return np.mean(pred,axis=0)

if __name__=='__main__':
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train'];x,y,tr=t['x'],t['y'],t['traj'];rng=np.random.default_rng(141)
 I=[np.arange(52),np.arange(16,72),np.r_[0:27,43:72],np.r_[0:40,56:72],np.arange(56)]+[np.sort(rng.choice(len(x),56,replace=False)) for _ in range(4)]
 E=[];P=[];Y=[];B=[]
 for i in I:
    j=np.setdiff1d(np.arange(len(x)),i);Y.append(y[j]);d={**s,'train':{k:t[k][i] for k in ('x','y','traj')},'test':{'x':x[j]}}
    B.append(solve_batch(d)['y_hat']); out={}
    for key,f in _library(s,x[np.r_[i,j]],y[i],tr[i]):
       for m,p,loo in _fits(f,y[i]):out[key,m]=p
    P.append(out)
 keys=list(set.intersection(*(set(p) for p in P)));keys.sort(key=repr)
 A=np.array([[np.mean(abs(p[k]-v)) for p,v in zip(P,Y)] for k in keys]);base=np.array([np.mean(abs(p-v)) for p,v in zip(B,Y)])
 print('base',base.round(5));print('best',[(keys[i],A[i].round(5)) for i in np.argsort(A.max(1))[:8]])
 # Compare families on all stress slices; do not fit any coefficients to validation labels.
 best=(base.max(),[],0,base)
 for alpha in (.5,1.,2.):
    score=A.mean(1)+alpha*A.max(1)
    for width in (3,6,12,24,48,96):
       order=[]; seen=set()
       for k in np.argsort(score):
          if keys[k][0] in seen:continue
          order.append(keys[k]); seen.add(keys[k][0]);
          if len(order)==width:break
       for weight in (.5,1.):
          e=np.array([np.mean(abs((1-weight)*b+weight*np.mean([p[k] for k in order],0)-v)) for b,p,v in zip(B,P,Y)])
          if e.max()<best[0]:best=(e.max(),order,weight,e)
 print('selected',len(best[1]),best[2],best[3].round(6));
 if best[1]:
    selected=[(k,[m]) for k,m in best[1]]
    src=open(__file__).read().split("if __name__==")[0].split('def _library',1)[1]
    path='/holdout_task/batch_solver.py'
    baseline=open(path).read(); baseline=baseline.replace('def solve_batch(study):','def _base_solve(study):');
    new='\n\ndef _library'+src+'\n_SELECTED = '+repr(selected)+'\n\ndef solve_batch(study):\n    out = _base_solve(study)\n    tr = study["train"]\n    x = np.concatenate((tr["x"], study["test"]["x"]))\n    y = '+repr(1-best[2])+' * out["y_hat"] + '+repr(best[2])+' * _predict(study, x, tr["y"], tr["traj"], _SELECTED)\n    out["y_hat"] = y\n    out["diagnostics_vec"][:2] = (y.mean(), y.std())\n    return out\n'
    open(path,'w').write(baseline+new)

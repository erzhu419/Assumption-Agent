import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import test_dag_candidate as test
import node_00 as node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; ids=np.arange(len(t['x'])); kernel=node._kernels

def augmented(groups,coords,n):
    z,h,b,c,d,x=coords
    extra=[]
    for u in (z,b,c,d):
        for j in range(u.shape[1]):
            a=u[:,j,None]; q=a/np.maximum(a[:n].std(),1e-6)
            for lo,hi in ((.5,2.5),(2.5,5.5),(5.5,10.5)):
                r=q*np.linspace(lo,hi,5); extra.append(np.c_[np.sin(r),np.cos(r)])
            for k in range(j):
                v=u[:,k,None]; v=v/np.maximum(v[:n].std(),1e-6)
                extra.append(np.c_[q*v,q*q*v,q*v*v,q*q*v*v])
                for w in (q+v,q-v):
                    extra.append(np.c_[np.sin(w),np.cos(w),np.sin(2*w),np.cos(2*w)])
    for u,v in ((z,b),(z,c),(z,d),(b,c),(b,d),(c,d)):
        for j in range(4):
            a=u[:,j,None]; q=v[:,j,None]; extra.append(np.c_[a*q,a*q*q,a*a*q,a/(1+.22*q*q),q/(1+.22*a*a)])
    # Preserve the panel C cubic terms in the profiled feature bank.
    p=s['panel_c']; u=p['u']; design=lambda a: node._design(a,a*np.roll(a,-1,axis=1),a**3)
    w=node._fit(design(u),p['v1']); cv=design(z)@w
    extra += [cv,np.tanh(cv),np.sin(1.25*cv),np.sin(2.15*cv),cv*z,cv*b,cv*d]
    return kernel(groups+extra,coords,n)

splits=np.array_split(np.random.default_rng(23).permutation(ids),4)
scores=[]
for mode in (kernel,augmented):
    node._kernels=mode
    fold=[]; horizons=[]
    for j in splits:
        i=np.setdiff1d(ids,j); y=t['y'][i]; e=t['traj'][i,-1]
        b=dict(s,train=dict(x=t['x'][i],y=y,traj=t['traj'][i],summary=np.array([y.mean(),y.std(),e.mean(),e.std()])),test=dict(x=t['x'][j])); r,_=test.run_dag(b)
        h=np.abs(r['state_pred']-t['traj'][j]); fold.append([np.abs(r['y_hat']-t['y'][j]).mean(),h.mean()]); horizons.append(h.mean(0))
    f=np.array(fold); print(mode.__name__, 'fold MAEs',f.tolist(),'worst',f.max(0).tolist(),'worst horizon',np.max(horizons,axis=0).tolist(),flush=True)
    scores.append(f)
a,b=scores
if np.all(b.max(0)<=a.max(0)+.003) and np.max(b.max(0))<np.max(a.max(0))-.003:
    import inspect
    text=inspect.getsource(augmented).replace('def augmented(groups,coords,n):','def _kernels(groups,coords,n,panel_c):').replace('node._design','_design').replace('node._fit','_fit').replace("p=s['panel_c'];",'p=panel_c;').replace('return kernel(groups+extra,coords,n)','return _base_kernels(groups+extra,coords,n)')
    path='/holdout_task/node_00.py'
    src=open(path).read().replace('def _kernels(groups, coords, n):','def _base_kernels(groups, coords, n):').replace('k=_kernels(g,z,n)',"k=_kernels(g,z,n,s['panel_c'])")
    open(path,'w').write(src+'\n\n'+text)
    print('Promoted the richer feature bank.',flush=True)
else:
    print('Retained the incumbent under the worst-fold rule.',flush=True)

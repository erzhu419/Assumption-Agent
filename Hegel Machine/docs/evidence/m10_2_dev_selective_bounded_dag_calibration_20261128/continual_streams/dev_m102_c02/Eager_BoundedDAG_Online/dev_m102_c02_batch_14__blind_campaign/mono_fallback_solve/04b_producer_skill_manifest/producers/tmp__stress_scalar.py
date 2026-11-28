import pickle,runpy,numpy as np
from scipy.optimize import least_squares
p='/holdout_task/batch_solver.py'; m=runpy.run_path(p)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; z,b,a=m['_build'](s,t['x']); y=t['y']; rng=np.random.default_rng(42)
orders=[np.arange(72),*[rng.permutation(72) for _ in range(3)],np.argsort(np.linalg.norm(t['x'],axis=1))]; orders += [np.argsort(np.linalg.norm(z,axis=1)),np.argsort(np.linalg.norm(b,axis=1))]
src='''def _shrink_scalar(a,b,y,strength):
    n=len(y)
    a=a/np.maximum(a[:n].std(0),.08)
    b=b/np.maximum(b[:n].std(0),.08)
    target=np.r_[y,np.zeros(a.shape[1]+1)]
    pen=np.diag(np.r_[0,np.full(4,.06*strength),np.full(a.shape[1]-5,strength),.035*strength])
    starts=np.r_[np.eye(4),np.random.default_rng(1).normal(0,.5,(8,4))]
    outs=[]
    for link in (np.sin,np.tanh):
        def design(p,base,q):
            return np.column_stack((base,link(p[0]+q@p[1:])))
        def residual(p):
            mat=np.vstack((design(p,a[:n],b[:n]),pen))
            return np.r_[mat@_ls(mat,target)-target,.008*p]
        opt=min((least_squares(residual,np.r_[.1,start],max_nfev=100) for start in starts),key=lambda o:o.cost)
        pred=design(opt.x,a,b)
        outs.append(pred[n:]@_ls(np.vstack((pred[:n],pen)),target))
    return np.mean(outs,axis=0)
'''
exec(src,m); shuffled=[np.r_[o[:52],o[52:]] for o in orders]
def preds(fun):return [fun(a[o],b[o],y[o[:52]]) for o in shuffled]
def errors(pr):return np.array([np.mean(abs(q-y[o[52:]])) for q,o in zip(pr,shuffled)])
P=preds(m['_scalar']); baseline=errors(P); print('base',baseline,flush=True); best=(baseline[4:].max(),None)
for strength in (.04,.12,.28,.55,1.):
 Q=preds(lambda a,b,y:m['_shrink_scalar'](a,b,y,strength))
 for mix in (.5,1.):
  e=errors([(1-mix)*p+mix*q for p,q in zip(P,Q)]); print(strength,mix,e,flush=True)
  if max(e[:4])<=.038 and e[:4].mean()<=baseline[:4].mean()+.002 and e[4:].max()<best[0]-.006: best=(e[4:].max(),(strength,mix))
if best[1]:
 strength,mix=best[1]; body=open(p).read(); body=body.replace('def solve_batch(study):',src+'\n\ndef solve_batch(study):').replace("y_hat=_scalar(a,b,train['y'])",f"y_hat={1-mix}*_scalar(a,b,train['y'])+{mix}*_shrink_scalar(a,b,train['y'],{strength})" if mix!=1 else f"y_hat=_shrink_scalar(a,b,train['y'],{strength})"); open(p,'w').write(body); print('promoted',best,flush=True)

import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares

def load(path):
    spec=importlib.util.spec_from_file_location('candidate',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

D='/holdout_task/'
s=pickle.load(open(D+'public_batch.pkl','rb'))[0]['study']
a=load(D+'_frozen_nodes/node_00.py').run_node({'__input__':s})
b=load(D+'_frozen_nodes/node_01.py').run_node({'node_00':a})
u,bv=a['u_tr'],b['b_tr']; y,tr,t=s['train']['y'],s['train']['traj'],s['eval_times']

def dic(u,bv=None):
    fs={'1':np.ones(len(u))}; v={f'u{i}':x for i,x in enumerate(u.T)}
    if bv is not None: v.update({f'b{i}':x for i,x in enumerate(bv.T)})
    fs.update(v)
    for i,(k,x) in enumerate(v.items()):
        for l,z in list(v.items())[i:]: fs[k+'*'+l]=x*z
        fs['tanh('+k+')']=np.tanh(x)
        fs['sin('+k+')']=np.sin(x)
        fs[k+'^3']=x**3
    return fs

def omp(f,z,n=14):
    names=list(f); X=np.column_stack(list(f.values())); norm=np.linalg.norm(X,axis=0); X=X/np.maximum(norm,1e-10); ids=[]; out=[]; r=z
    for i in range(n):
        scores=abs(X.T@r); scores[ids]=-1; j=np.argmax(scores); ids.append(j); w=np.linalg.lstsq(X[:,ids],z,rcond=None)[0]; r=z-X[:,ids]@w; out.append((names[j],round(np.mean(abs(r)),5)))
    return out

def stats(f,z):
    X=np.column_stack(list(f.values())); w=np.linalg.lstsq(X[:52],z[:52],rcond=None)[0]; return np.mean(abs(X[52:]@w-z[52:]),axis=0)

if __name__=='__main__':
 print('time',t,'svd',np.linalg.svd(tr,compute_uv=False),'mean',tr.mean(0),'std',tr.std(0)); print('calibrated ranges',u.min(0),u.max(0),'B ranges',bv.min(0),bv.max(0))
 for p in ('panel_b','panel_c'):
    U=s[p]['u']; f=dic(U); print(p)
    subsets=[['1']+[f'u{i}' for i in range(4)],['1']+[f'u{i}' for i in range(4)]+[f'u{i}*u{j}' for i in range(4) for j in range(i,4)]]
    for key in s[p]:
      if key=='u': continue
      V=s[p][key]; print(key, 'linear/quadratic panel residuals', [np.mean(abs((X:=np.column_stack([f[n] for n in names]))@np.linalg.lstsq(X,V,rcond=None)[0]-V)) for names in subsets]); print('first column OMP',omp(f,V[:,0],9))
 U=s['panel_b']['u']; V=s['panel_b']['v']; fit=lambda k:np.column_stack([np.ones(len(U)),U,np.sin(U*k)]); fun=lambda k: (fit(k)@np.linalg.lstsq(fit(k),V,rcond=None)[0]-V).ravel(); res=least_squares(fun,np.full(4,2.8),bounds=(.35,6)); print('B frozen frequency',res.x,'B MAE',np.mean(abs(res.fun)))
 f=dic(u,bv)
 for i,z in enumerate([y,tr[:,0],tr[:,-1],tr[:,-1]-tr[:,0]*3]):
    print('OMP',i,omp(f,z)); print('MAEs linear, u quadratic, u*b, full quadratic, expanded', [stats({k:f[k] for k in ns},z) for ns in [list(f)[:9],['1']+[f'u{j}' for j in range(4)]+[f'u{j}*u{k}' for j in range(4) for k in range(j,4)],list(f)[:9]+[f'u{j}*b{k}' for j in range(4) for k in range(4)], [k for k in f if '(' not in k and '^' not in k], list(f)]])
 def temporal(z,kind):
    def des(k):
      q=-np.expm1(-k*t); return (q[:,None] if kind==0 else np.column_stack([q,t]) if kind==1 else np.column_stack([q,np.exp(-k*t)]) if kind==2 else np.column_stack([q,-np.expm1(-k[1]*t)]))
    if kind==3:
      def des(k): return -np.expm1(-t[:,None]*k)
    def err(k):
      A=des(k); return A@np.linalg.lstsq(A,z,rcond=None)[0]-z
    k=least_squares(err,[.5,2.] if kind==3 else [.7],bounds=(.02,8)).x
    return k,np.sqrt(np.mean(err(k)**2))
 for kind in range(4):
    vals=[temporal(row,kind) for row in tr]; print('temporal',kind,'rate min,med,max',np.percentile([r for r,e in vals],[0,50,100],axis=0),'rmse med,max',np.percentile([e for r,e in vals],[50,100]))

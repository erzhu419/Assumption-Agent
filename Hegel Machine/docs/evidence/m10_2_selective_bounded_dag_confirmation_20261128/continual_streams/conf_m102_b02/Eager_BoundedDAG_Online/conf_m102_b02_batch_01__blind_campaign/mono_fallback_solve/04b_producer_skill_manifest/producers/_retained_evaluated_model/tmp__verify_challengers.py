import importlib.util,pickle,numpy as np
from scipy.linalg import cho_factor,cho_solve
p=importlib.util.spec_from_file_location('candidate','/tmp/baseline_batch.py'); m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];z,b=m._coordinates(s,tr['x']);u=s['panel_b']['u'];v=s['panel_b']['v']; ba=m._design(z)@m._fit(m._design(u),v); n=len(z);rng=np.random.default_rng(20531);folds=[np.arange(52,n)]+[rng.choice(n,20,replace=False) for _ in range(40)];
# Also hold out the largest calibrated radii, an extrapolation stress test.
folds.append(np.argsort(np.linalg.norm(z,axis=1))[-20:]);
def predict(i,j,bc,kind,ss,bb,w,reg):
 def sat(h):return np.tanh(h) if kind=='tanh' else h/np.sqrt(1+h*h)
 x=np.c_[z,bc,sat(ss*z),sat(bb*bc)];x/=np.maximum(x[i].std(0),1e-10);a=x[i];te=x[j];
 def ker(x,q):
  S=(1+x@q.T/x.shape[1])**2;return S+w*(m._kernel(x,q)-S)
 T=tr['traj'][i];modes=np.linalg.svd(T,full_matrices=False)[2][:2];Y=np.c_[tr['y'][i],T@modes.T];K=ker(a,a);pred=ker(te,a)@cho_solve(cho_factor(K+reg*np.eye(len(i))),Y);return np.c_[pred[:,0],pred[:,1:]@modes]
P={}; pars=[('baseline',b,'tanh',2,2,1,.001),('tanh',b,'tanh',1,2.5,.15,.004),('rat',b,'rat',1,2,1,.004),('rat25',b,'rat',1,2.5,.15,.004),('affine',ba,'tanh',1,2.5,.15,.004)];
for name,*p in pars:P[name]=[predict(np.setdiff1d(np.arange(n),j),j,*p) for j in folds]
P['ensemble']=[(a+b+c)/3 for a,b,c in zip(P['tanh'],P['rat'],P['rat25'])]; P['balanced']=[.5*a+.25*b+.25*c for a,b,c in zip(P['tanh'],P['rat'],P['rat25'])]
for name,p in P.items():
 vals=[]
 for pred,j in zip(p,folds):
  d=abs(pred-np.c_[tr['y'][j],tr['traj'][j]]);vals.append([d[:,0].mean(),d[:,1:].mean(),d[:,-1].mean(),d[:,1:].mean(0).max()]);
 a=np.array(vals);print(name,'52/20',np.round(a[0],5),'fresh mean',np.round(a[1:-1].mean(0),5),'fresh max',np.round(a[1:-1].max(0),5),'large radius',np.round(a[-1],5))

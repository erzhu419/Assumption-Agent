import pickle,numpy as np,sys,itertools
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task/_frozen_nodes');import node_00 as f
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; A=f._affine(u); x=f._inverse(s['train']['x'],f._fit_sensor(s['panel_a'])); y=s['train']['y']; tr=s['train']['traj']; U,D,V=np.linalg.svd(tr,full_matrices=False); H=np.c_[y,U[:,:2]*D[:2]]
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def poly(x,k):
 return np.column_stack([np.prod(x[:,j],axis=1) for n in range(k+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),n)])
def cv(A,H,ids=None):
 if ids is None:ids=np.arange(len(A))
 i=ids[:-20];j=ids[-20:];return np.round(abs(A[j]@fit(A[i],H[i])-H[j]).mean(0),5)
def aff(x):return np.c_[np.ones(len(x)),x]
def bbase(q):return np.c_[aff(q),np.cos(2.6*q)]
def cbase(q):return np.c_[aff(q),np.tanh(2.15*q)]
def dbase(q):return np.c_[aff(q),np.cos(1.25*q)]
p=s['panel_b'];pc=s['panel_c'];z=bbase(x)@fit(bbase(p['u']),p['v']); c=cbase(x)@fit(cbase(pc['u']),pc['v1']); d=dbase(x)@fit(dbase(pc['u']),pc['v2']);
from scipy.special import erf
from scipy.optimize import least_squares
L=aff(z); bil=z[:,0]*z[:,2]+z[:,1]*z[:,3]; P=poly(z,3); Q=np.c_[z[:,0]*z[:,2],z[:,1]*z[:,3]]
shapes={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'erf':erf,'soft':lambda t:t/np.sqrt(1+t*t)}
allidx=np.arange(len(z)); folds=[(allidx[:-20],allidx[-20:])]+[(np.setdiff1d(allidx,j),j) for seed in (42,142,242) for j in np.array_split(np.random.default_rng(seed).permutation(len(z)),5)]
print('cubic',np.round([np.mean(abs(P[j]@fit(P[i],y[i])-y[j])) for i,j in folds],5))
for name,f in shapes.items():
 for mode in ['sum','pairs','inside','linear']:
  def pred(p,i):
   qi=Q[i] if mode=='pairs' else bil[i,None]; ts=L[i]@p[:5]; b=qi@p[6:6+qi.shape[1]]
   if mode=='inside':return p[5]*f(ts+b)
   if mode=='linear': b=b+p[-1]*ts
   return p[5]*f(ts)+b
  def fitp(i):
   n=2 if mode=='pairs' else 1; init=np.r_[fit(L[i],y[i])/2.2,2.2,np.full(n,.22),([0.] if mode=='linear' else [])]; lower=np.r_[np.full(5,-10.),.25,np.full(len(init)-6,-4.)]; upper=np.r_[np.full(5,10.),12.,np.full(len(init)-6,4.)]
   return least_squares(lambda p:pred(p,i)-y[i],init,bounds=(lower,upper),max_nfev=300).x
  p=fitp(allidx); scores=np.array([np.mean(abs(pred(fitp(i),j)-y[j])) for i,j in folds]);print(name,mode,'fit',round(np.mean(abs(pred(p,allidx)-y)),6),'public/mean/worst',np.round([scores[0],scores[1:].mean(),scores.max()],6),'coef',np.round(p,4))

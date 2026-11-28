import pickle,numpy as np,itertools
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; Y=np.c_[t['y'],t['traj']]; x=t['x'];
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def score(f):
 w=ls(f[:52],Y[:52]); e=abs(f@w-Y); return np.round([e[:52,0].mean(),e[52:,0].mean(),e[:52,1:].mean(),e[52:,1:].mean()],5)
d=s['panel_a']; u,v=d['u'],d['v']; good=np.max(abs(v-u),axis=1)<3; print('good',np.where(~good));
def aterms(q,j,shift):
 a,b=q[:,j],q[:,(j+shift)%4]; return np.c_[a,b,a*b,a**3]
for k in [1,-1,2]:
 ws=np.array([ls(aff(aterms(u,j,k))[good],v[good,j]) for j in range(4)]).T
 print('A',k,ws, 'mae',np.mean(abs(np.column_stack([aff(aterms(u,j,k))[good]@ws[:,j] for j in range(4)])-v[good])))
 if k==1:
  z=np.array([least_squares(lambda q: np.array([aff(aterms(q[None],j,k))[0]@ws[:,j] for j in range(4)])-r,r).x for r in x]); Az=lambda x:np.column_stack([aff(aterms(x,j,1))@aw[:,j] for j in range(4)]); aw=ws.copy()
d=s['panel_b']; w=ls(aff(d['u']),d['v']); bv=lambda q:aff(q)@w
for label,q in [('x',x),('invA',z),('A',Az(x)),('B(x)',bv(x)),('B(invA)',bv(z))]:
 print(label);print('polynomial',[(i,score(PolynomialFeatures(i).fit_transform(q))) for i in (1,2,3)]);
 for fun in [np.tanh,np.sin,lambda a: a/(1+abs(a)), lambda a:a*a/(1+a*a),lambda a:np.log1p(a*a)]:
  print(score(aff(np.c_[q,fun(q)])),score(PolynomialFeatures(2).fit_transform(fun(q))))
# extract shared rates
time=s['eval_times']; h=lambda r:-np.expm1(-time[:,None]*r)
r=least_squares(lambda r:(h(r)@ls(h(r),Y[:52,1:].T)-Y[:52,1:].T).ravel(),[1.8,.42],bounds=([.8,.05],[4,1.1])).x
amp=ls(h(r),Y[:,1:].T).T; print('rates',r,'amp correlations',np.corrcoef(np.c_[x,z,bv(z),Y[:,0],amp].T)[-3:,:].round(3))
np.savez('/tmp/exp.npz',x=x,z=z,amp=amp,rate=r,w=w)

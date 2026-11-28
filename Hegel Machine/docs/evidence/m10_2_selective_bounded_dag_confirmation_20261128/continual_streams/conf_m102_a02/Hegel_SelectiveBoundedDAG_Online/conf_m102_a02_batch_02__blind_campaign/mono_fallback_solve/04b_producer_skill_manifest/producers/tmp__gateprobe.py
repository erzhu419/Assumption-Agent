import pickle,numpy as np
from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
poly=lambda x,d:PolynomialFeatures(d).fit_transform(x)
p=s['panel_a']; u,v=p['u'],p['v']; m=np.linalg.norm(u-v,axis=1)<1
A=np.array([ls(poly(u[m,i:i+1],1),v[m,i]) for i in range(4)]); af=lambda x:x*A[:,1]+A[:,0]; ai=lambda x:(x-A[:,0])/A[:,1]
p=s['panel_b']; B=ls(poly(p['u'],1),p['v']); bf=lambda x:poly(x,1)@B
from scipy.optimize import least_squares
from sklearn.model_selection import ShuffleSplit
q=ai(x); Q=poly(q,2); ids=np.arange(72); pairs=[(i,j) for i in range(4) for j in range(i,4)]
splits=[(ids[:52],ids[52:]),*KFold(4,shuffle=True,random_state=12).split(x),*KFold(4,shuffle=True,random_state=73).split(x),*ShuffleSplit(32,test_size=20,random_state=383).split(x)]
Qix={'all':np.arange(15),'gate2':np.array([0,1,2,3,4,*[5+k for k,(i,j) in enumerate(pairs) if i==2 or j==2]]),'linear':np.arange(5),'gate3':np.array([0,1,2,3,4,*[5+k for k,(i,j) in enumerate(pairs) if i==2 or j==2 or (i,j) in [(0,0),(1,1),(3,3)]]])}
def gate(par):return np.tanh(par[0]*(q[:,2]-par[1])+q[:,[0,1,3]]@par[2:])
def mat(par,ix):return np.c_[Q,Q[:,ix]*gate(par)[:,None]]
def bet(F,y,lam):
 if not lam:return ls(F,y)
 diag=np.full(F.shape[1],lam);diag[:5]=lam/100;return np.linalg.solve(F.T@F+np.diag(diag),F.T@y)
def setup(a,ix,lam,mode):
 if isinstance(mode,(int,float)):return mat([mode,0,0,0,0],ix)
 n={'slope':1,'center':2,'tilt':5}[mode]
 def obj(p):
  F=mat(np.r_[p,np.zeros(5-n)],ix)[a]; return F@bet(F,y[a],lam)-y[a]
 p=least_squares(obj,[8,0,0,0,0][:n],bounds=([1,-.35,-3,-3,-3][:n],[35,.35,3,3,3][:n]),max_nfev=50).x
 return mat(np.r_[p,np.zeros(5-n)],ix)
rows=[]
for n,ix in Qix.items():
 for lam in [0,.0003,.003,.015,.06]:
  for mode in [4,8,12,18,'slope','center','tilt']:
   es=[]
   for a,b in splits:
    F=setup(a,ix,lam,mode);es.append(np.mean(abs(F[b]@bet(F[a],y[a],lam)-y[b])))
   rows.append((np.round([es[0],np.mean(es[1:9]),max(es[1:9]),np.mean(es[9:]),max(es[9:])],6),n,lam,mode))
print('baseline',[r for r in rows if r[1:]==('all',0,8)]); print('best',*sorted(rows,key=lambda r:(r[0][4],r[0][3]))[:25],sep='\n');print('best mean',*sorted(rows,key=lambda r:r[0][3])[:15],sep='\n')
F=mat([8,0,0,0,0],Qix['all']); beta=ls(F,y);print('gate coeffs',np.round(beta.reshape(2,-1),4));print('sorted residual',np.round(np.c_[q[:,2], y-F@beta][np.argsort(q[:,2])],4))

import numpy as np, pickle
from scipy.optimize import root
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; y=s['train']['y']; P=s['train']['traj']; x=s['train']['x']
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def ab(u,j,deg):
 a=u[:,j]; b=u[:,(j+1)%4]; c=u[:,(j-1)%4]; return np.c_[np.ones(len(a)),a,b,c,a*b,a*c,a**3,a*a,b*b] if deg else aff(u)
a=s['panel_a']; u,v=a['u'],a['v']; fits=[]
for deg in [0,1]:
 def fit(j):
  d=ab(u,j,deg); keep=np.arange(len(u))
  for _ in range(15):
   w=ls(d[keep],v[keep,j]); r=abs(d@w-v[:,j]); keep=np.argsort(r)[:36]
  keep=r<max(.001,8*np.median(r)); return ls(d[keep],v[keep,j]),np.round(np.sort(r)[::4],5)
 fits.append([fit(j)[0] for j in range(4)]);print('cal',deg,[fit(j) for j in range(4)])
def poly(x,p):return np.column_stack([np.prod(x[:,jj],axis=1) for i in range(1,p+1) for jj in cr(range(x.shape[1]),i)])
def coord(x):
 z=np.array([root(lambda q:np.array([ab(q[None],j,1)[0]@fits[1][j] for j in range(4)])-r,r).x for r in x]); b,c=s['panel_b'],s['panel_c']; w=aff(z)@ls(aff(b['u']),b['v']); out={'x':x,'z':z,'w':w};
 for mode,fun in [('quad',lambda x:poly(x,2)),('cross',lambda x:poly(x,2)[:,[5,6,7,9,10,12]]),('even',lambda x:x*x)]:
  for lab in ['v1','v2']:
   D=aff(fun(c['u'])); cf=ls(D,c[lab]); print('C',mode,lab,np.mean(abs(D@cf-c[lab])),np.round(cf,3) if mode=='cross' else '')
   for key,r in [('w',w),('z',z)]:out[key+mode+lab]=aff(fun(r))@cf
 return out
if __name__=='__main__':
 print('svd',np.linalg.svd(P,compute_uv=False)); print('ranges',x.min(0),x.max(0))
 out=coord(x)
 for k,z in out.items():
  for p in [1,2,3]:
   d=aff(poly(z,p)); r=d@ls(d[:52],np.c_[y,P][:52])-np.c_[y,P];print(k,p,np.round([abs(r[:52,0]).mean(),abs(r[52:,0]).mean(),abs(r[:52,1:]).mean(),abs(r[52:,1:]).mean()],4))
 print('traj PCA',np.linalg.svd(P-P.mean(0),full_matrices=False)[2][:3]);print('ratios',np.c_[P[:,[0,3,7]],P[:,0]/P[:,7]][np.argsort(P[:,7])[::4]])

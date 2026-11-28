import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=s['panel_a']['u'],s['panel_a']['v']; m=np.linalg.norm(v-u,axis=1)<5; u,v=u[m],v[m]; w=np.roll(u,2,axis=1)
f={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'cubic':lambda x:x**3,'septic':lambda x:x**7,'soft':lambda x:x/np.sqrt(1+x*x)}
def solve(a,b,names,verbose=False):
 F,G=map(f.get,names)
 def design(p,j):return np.c_[np.ones(len(a)),a[:,j],F(p[0]*a[:,j]),G(p[1]*b[:,j])]
 def calc(p):
  ds=[design(p,j) for j in range(4)]; cs=[np.linalg.lstsq(d,v[:,j],rcond=None)[0] for j,d in enumerate(ds)]; return np.column_stack([d@c for d,c in zip(ds,cs)])-v,cs
 opt=[least_squares(lambda p:calc(p)[0].ravel(),[a,b],bounds=([.15,.15],[6.,6.]),max_nfev=90) for a,b in [(1.,1.),(1.7,2.15),(3.1,3.1)]]; z=min(opt,key=lambda r:r.cost); e,c=calc(z.x); return np.sqrt(np.mean(e**2)),names,z.x,np.array(c),np.max(abs(e))
for row in sorted([solve(u,w,(a,b)) for a in f for b in f],key=lambda t:t[0])[:12]:print(row)
print('univariate polynomial in opposing pair:');
for j in range(4):
 a,b=u[:,j],w[:,j]
 for deg in [5,7,9]:
  D=np.c_[np.ones(len(a)),*[x**k for k in range(1,deg+1,2) for x in (a,b)]]; c=np.linalg.lstsq(D,v[:,j],rcond=None)[0]; print(j,deg,c,np.std(D@c-v[:,j]))

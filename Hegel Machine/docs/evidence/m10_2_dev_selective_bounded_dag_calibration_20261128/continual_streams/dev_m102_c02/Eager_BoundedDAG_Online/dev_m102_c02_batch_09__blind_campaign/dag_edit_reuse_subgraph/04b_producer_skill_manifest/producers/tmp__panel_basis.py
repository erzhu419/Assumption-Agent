import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.max(abs(v),axis=1)<3; u,v=u[keep],v[keep]
q=u[:,[2,3,0,1]]
fns={'sin':np.sin,'tanh':np.tanh,'cubic':lambda a:a**3,'rat':lambda a:a/(1+a*a),'atan':np.arctan,'sinh':np.sinh}
np.set_printoptions(precision=6,suppress=True)
for lin_cross in (False,True):
 ans=[]
 for own,other in itertools.product(fns,repeat=2):
  def calc(f,ids,details=False):
   E=[]; B=[]
   for j in range(4):
    a=u[ids,j]; b=q[ids,j]; A=np.column_stack([np.ones(len(a)),a]+([b] if lin_cross else [])+[fns[own](f[0]*a),fns[other](f[1]*b)]); y=v[ids,j]; co=np.linalg.lstsq(A,y,rcond=None)[0]; E.append(A@co-y); B.append(co)
   return (np.array(E),np.array(B)) if details else np.ravel(E)
  ids=np.arange(len(u)); starts=([1,2.15],[2,2.5],[3,3],[1,1]); fit=min((least_squares(calc,start,args=(ids,),bounds=(.15,5),max_nfev=55) for start in starts),key=lambda r:r.cost)
  E,B=calc(fit.x,ids,True); ans.append((np.sqrt(np.mean(E**2)),own,other,fit.x,B))
 print('linear cross',lin_cross)
 for rms,own,other,f,B in sorted(ans,key=lambda a:a[0])[:6]: print(rms,own,other,f,'coef',B)

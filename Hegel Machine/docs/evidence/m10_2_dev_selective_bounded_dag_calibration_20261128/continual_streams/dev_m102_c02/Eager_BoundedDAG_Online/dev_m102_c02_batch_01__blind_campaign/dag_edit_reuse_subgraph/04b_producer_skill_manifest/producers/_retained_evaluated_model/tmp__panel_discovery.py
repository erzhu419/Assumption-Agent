import pickle,itertools,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
tr=s['train']; u,v=[s['panel_a'][k] for k in ('u','v')]; a=[]
for x,y in zip(u.T,v.T):
 i,j=np.triu_indices(len(x),1); d=x[j]-x[i]; m=abs(d)>.2; g=np.median((y[j[m]]-y[i[m]])/d[m]); b=np.median(y-g*x); keep=abs(y-b-g*x)<.01; a.append(np.linalg.lstsq(np.c_[np.ones(sum(keep)),x[keep]],y[keep],rcond=None)[0])
b,g=np.array(a).T; x=(tr['x']-b)/g
funs={'sin':lambda x,k:np.sin(k*x), 'tanh':lambda x,k:np.tanh(k*x),'rat1':lambda x,k:x/(1+k*x*x),'rat2':lambda x,k:x/(1+k*x*x)**.5,'sinh':lambda x,k:np.sinh(k*x),'cos':lambda x,k:np.cos(k*x)-1,'cosh':lambda x,k:np.cosh(k*x)-1,'ratE':lambda x,k:x*x/(1+k*x*x),'expE':lambda x,k:np.expm1(-k*x*x),'poly3':lambda x,k:x**3,'poly2':lambda x,k:x*x,'poly4':lambda x,k:x**4,'shift-sin':lambda x,k:np.sin(k*x+.5),'shift-tanh':lambda x,k:np.tanh(k*x+.5)}
def design(x,f,k,lin):return np.column_stack([np.ones(len(x))]+([x] if lin else [])+[f(x,k)])
allmaps=[]
for p,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,y=s[p]['u'],s[p][j]; res=[]
 for name,f in funs.items():
  for lin in [0,1]:
   def get(k):
    A=design(u,f,k,lin); inv=np.linalg.pinv(A); r=y-A@inv@y; h=np.sum(A*inv.T,axis=1); return np.mean((r/(1-h[:,None]))**2)**.5,inv@y
   grid=np.linspace(.035,6,81); ii=np.argmin([get(k)[0] for k in grid]); opt=minimize_scalar(lambda k:get(k)[0],bounds=(grid[max(ii-1,0)],grid[min(ii+1,len(grid)-1)]),method='bounded'); k=opt.x; err,B=get(k); pred=design(x,f,k,lin)@B; res.append((err,name,lin,k,pred))
 res.sort(key=lambda e:e[0]); print(p,j,[(round(e,6),n,l,round(k,5)) for e,n,l,k,_ in res[:8]],flush=True); allmaps.append(res)
def cv(c):
 des=np.c_[np.ones(len(c)),c]; dy=np.c_[des,(c[:,:2]*c[:,2:4]).sum(1)]; i=52
 return tuple(np.mean(abs(A[i:]@np.linalg.lstsq(A[:i],z[:i],rcond=None)[0]-z[i:])) for A,z in ((dy,tr['y']),(des,tr['traj'])))
for choice in itertools.product(*(r[:3] for r in allmaps)):
 B,C,D=[r[-1] for r in choice]; models=[B,C,D,B+C,B+D,C+D,B+C+D,B+.22*C+.22*D,.22*B+C+.22*D,.22*B+.22*C+D,B+.22*C,B+.22*D,C+.22*D,B-.22*C-.22*D,B+.22*C-.22*D,B-.22*C+.22*D,B+.22*x+.22*C+.22*D]
 if choice[0] is allmaps[0][0] and choice[1] is allmaps[1][0] and choice[2] is allmaps[2][0]:print('best maps combinations',[(i,np.round(cv(m),5)) for i,m in enumerate(models)])
 for i,m in enumerate(models):
  r=cv(m)
  if max(r)<.065: print('good',i,r,[q[1:4] for q in choice])
print('concatenated',cv(np.column_stack([r[0][-1] for r in allmaps]+[x])))

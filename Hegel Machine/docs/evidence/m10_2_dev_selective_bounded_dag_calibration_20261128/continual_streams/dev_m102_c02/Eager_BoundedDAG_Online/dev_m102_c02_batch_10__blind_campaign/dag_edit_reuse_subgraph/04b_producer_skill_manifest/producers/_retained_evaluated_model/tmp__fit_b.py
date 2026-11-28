import pickle,numpy as np
from scipy.optimize import minimize_scalar, least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
print('train/test x ranges:',[(s[k]['x'].min(),s[k]['x'].max()) for k in ('train','test')]); print('telemetry:',s['telemetry_vec'])
def regress(d,y):
 h=np.linalg.pinv(d,rcond=1e-10); w=h@y; r=d@w-y; return np.sqrt(np.mean((r/np.maximum(1-(d*h.T).sum(1)[:,None],1e-8))**2)),np.sqrt(np.mean(r**2)),w
u,v=[s['panel_a'][k] for k in ('u','v')]; good=(abs(v)<3).all(1); x,y=u[good],v[good]
def aa(k):
 rs=[]; ws=[]
 for j in range(4):
  q=[j,(j+2)%4]; d=np.c_[np.ones(len(x)),x[:,q],np.sin(k*x[:,q])]; loo,fit,w=regress(d,y[:,j:j+1]);rs.append((loo,fit));ws.append(w.ravel())
 return np.array(rs),np.array(ws)
ka=minimize_scalar(lambda k: np.mean(aa(k)[0][:,1]**2),bounds=(1.9,3.1),method='bounded').x
fs={'x':lambda k,x:x,'sin':lambda k,x:np.sin(k*x),'cos':lambda k,x:np.cos(k*x),'tanh':lambda k,x:np.tanh(k*x),'rat':lambda k,x:x/(1+k*x*x),'cube':lambda k,x:x**3,'q':lambda k,x:x*x,'atan':lambda k,x:np.arctan(k*x),'erf':lambda k,x:__import__('scipy').special.erf(k*x)}
from itertools import combinations_with_replacement
for name,key in [('panel_b','v')]:
 x,y=s[name]['u'],s[name][key]; ans=[]
 def details(a,b,p,tied):
  f,g=fs[a](p[0],x),fs[b](p[1],x); z=[f+p[2]*g] if tied else [f,g]; return regress(np.column_stack([np.ones(len(x)),*z]),y)
 for a,b in combinations_with_replacement(fs,2):
  if a==b and a in ('x','cube','q'):continue
  for tied in (True,False):
   for start in ([1,2.5,0.5],[1.25,1.25,-.6],[2,3,-.5]):
    if not tied and start[0]==start[1]:continue
    opt=least_squares(lambda p: np.ravel(details(a,b,p,tied)[1]),start,bounds=([.1,.1,-30],[8,8,30]),max_nfev=90,gtol=1e-12,ftol=1e-10)
    loo,fit,w=details(a,b,opt.x,tied);ans.append((loo,fit,a,b,tied,opt.x,w))
 print(name,key)
 for r in sorted(ans,key=lambda r:r[0])[:10]:print(r[:6],flush=True)
 import json; best=min(ans,key=lambda r:r[0]); json.dump([best[2],best[3],best[4],best[5].tolist()],open("/tmp/panel_b_best.json","w"))

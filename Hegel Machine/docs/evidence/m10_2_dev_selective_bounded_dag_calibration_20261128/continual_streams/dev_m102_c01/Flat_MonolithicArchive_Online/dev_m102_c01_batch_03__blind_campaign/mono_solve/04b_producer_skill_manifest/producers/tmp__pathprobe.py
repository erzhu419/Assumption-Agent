import numpy as np,pickle
from scipy.optimize import minimize_scalar
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from arc import _calibration
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; n=len(x)
a,b=_calibration(s['panel_a']); p=s['panel_b']; M=np.linalg.lstsq(np.c_[np.ones(len(p['u'])),p['u']],p['v'],rcond=None)[0]; B=lambda u:u@M[1:]+M[0]; Bi=lambda u:(u-M[0])@np.linalg.inv(M[1:]); p=s['panel_c']; u=p['u'];
fs={'tanh':np.tanh,'atan':np.arctan,'erf':__import__('scipy.special',fromlist=['erf']).erf,'root':lambda q:q/np.sqrt(1+q*q),'rat':lambda q:q/(1+q*q),'rat2':lambda q:q*q/(1+q*q),'sin':np.sin,'asinh':np.arcsinh}
C=[]
for k in ('v1','v2'):
 out=[]
 for key,f in fs.items():
  for lin in (0,1):
   def fit(ks):
    F=np.c_[np.ones(len(u)),u if lin else u[:,:0],f(u*ks)]; W=np.linalg.lstsq(F,p[k],rcond=None)[0]; return np.mean((F@W-p[k])**2),W
   from scipy.optimize import minimize
   r=minimize(lambda w:1e5*fit(w)[0],np.repeat(1.,4),bounds=[(.05,4)]*4,method='L-BFGS-B',options={'ftol':1e-14}); e,w=fit(r.x); out.append((e,key,lin,r.x,w))
 out.sort(key=lambda r:r[0]); print(k,[(e**.5,key,lin,k) for e,key,lin,k,w in out[:6]],flush=True); _,key,lin,k,w=out[0]; C.append(lambda q,f=fs[key],k=k,w=w,lin=lin:np.c_[np.ones(len(q)),q if lin else q[:,:0],f(q*k)]@w)

def paths(x):
  d={}
  for k,v in [('x',x),('z',(x-b)/a),('a',x*a+b)]:
   for name,f in [('',lambda u:u),('B',B),('I',Bi)]:
    q=f(v); N=name+k; d[N]=q
    for j,c in enumerate(C):
     w=c(q); d[str(j)+N]=w; d['B'+str(j)+N]=B(w); d['I'+str(j)+N]=Bi(w)
  return d
D=paths(x); ids=np.arange(n); splits=[(ids[:52],ids[52:])]+[(np.setdiff1d(ids,j),j) for j in np.array_split(np.random.default_rng(74).permutation(n),4)];
if __name__=='__main__':
 for k,qq in D.items():
  q=(qq-qq.mean(0))/np.sqrt(np.mean(qq.std(0)**2)); E=cdist(q,q,'sqeuclidean')
  for length in (.7,1,1.4,2,3,4):
   for ridge in (1e-6,1e-4,.002,.02):
    for pow in (0,1,2):
     base=1+q@q.T/4; K=np.exp(-.5*E/length**2)+(base**pow if pow else 1); es=[]
     for i,j in splits:
      A=solve(K[np.ix_(i,i)]+ridge*np.eye(len(i)),np.c_[y,t][i],assume_a='pos'); pred=K[np.ix_(j,i)]@A; es.append([np.mean(abs(pred[:,0]-y[j])),np.mean(abs(pred[:,1:]-t[j]))]);
     if k==next(iter(D)) and (length,ridge,pow)==(.7,1e-6,0):results=[]
     results.append((np.array(es),(k,length,ridge,pow)))
 for col in (0,1):
  for mode in (0,1):
   print('ranking',col,mode,[(tuple(np.round(e[:,col],5)),k) for e,k in sorted(results,key=lambda r:(r[0][0,col] if mode==0 else np.mean(r[0][:,col])))[:12]])
 np.save('/tmp/pathres.npy',np.array(results,dtype=object),allow_pickle=True)

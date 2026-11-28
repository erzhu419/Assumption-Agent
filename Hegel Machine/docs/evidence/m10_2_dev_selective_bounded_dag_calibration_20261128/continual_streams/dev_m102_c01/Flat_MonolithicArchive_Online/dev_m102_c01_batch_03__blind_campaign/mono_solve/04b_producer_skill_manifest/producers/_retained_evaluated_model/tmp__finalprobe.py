import sys,io,contextlib
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch,_fit
with contextlib.redirect_stdout(io.StringIO()): from pathprobe import *
from scipy.linalg import solve
rows=[]
base=[]
for i,j in splits:
 ss=dict(s,train={k:v[i] if k!='summary' else v for k,v in s['train'].items()},test={'x':x[j]}); base.append(solve_batch(ss)['y_hat'])
for nm in ['z','Bz','Iz']:
 v=D[nm]; eye=np.eye(4); a,b=np.triu_indices(4,1)
 for sign in (1,-1,0):
  f=v@np.c_[eye,(eye[:,a]+sign*eye[:,b])/np.sqrt(2)] if sign else v
  for scale in [0.9,1.2,1.5,1.8,2.1,2.4]:
   r=f*scale
   for key in ('spectral','tanh','atan'):
    F=np.c_[np.sin(r),np.cos(r)] if key=='spectral' else (np.tanh(r) if key=='tanh' else np.arctan(r)); U=F@F.T/f.shape[1]
    for order in (1,2,3):
     for h in (4,10):
      E=cdist(D['z'],D['z'],'sqeuclidean'); Z=D['z']; K=(1+U)**order+np.exp(-E/h)+1+Z@Z.T/4
      for noise in (1e-6,.0001,.001):
       P=[]; L=[]
       for i,j in splits:
        ids=np.r_[i,j]; p,e=_fit(K[np.ix_(ids,ids)],len(i),y[i],noise); P.append(p); L.append(np.mean(abs(e)))
       rows.append(((nm,sign,scale,key,order,h,noise),P,L))
def err(p): return np.array([np.mean(abs(a-y[j])) for a,(_,j) in zip(p,splits)])
print('base',err(base)); out=[]
for par,P,_ in rows:
 for alpha in (.5,.65,.8,1.):
  es=err([alpha*p+(1-alpha)*b for p,b in zip(P,base)]); out.append((es,par,alpha))
for e,pa,a in sorted([r for r in out if r[0][0]<=.038],key=lambda r:r[0].max()+.3*r[0].mean())[:16]:print('single',np.round(e,5),pa,a)
for nm,sign,key in [('Iz',1,'spectral'),('Iz',-1,'tanh'),('Iz',-1,'atan'),('z',1,'spectral')]:
 cand=[(P,L) for (n,s,g,k,o,h,r),P,L in rows if n==nm and s==sign and k==key and ((key=='spectral' and 1.5<=g<=2.1 and o<=2) or (key!='spectral' and 1.2<=g<=1.5 and o>=2)) and r<=.0001];
 for a in (0,.2,.4):
  P=[(1-a)*np.average(np.array([p[i] for p,l in cand]),axis=0,weights=np.exp(-np.array([l[i] for _,l in cand])/.01))+a*base[i] for i in range(5)]; print('family',nm,sign,key,a,err(P));

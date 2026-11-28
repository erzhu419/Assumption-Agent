import pickle,numpy as np
from scipy.optimize import minimize_scalar,minimize
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v')); keep=np.linalg.norm(v,axis=1)<6;u,v=u[keep],v[keep]; r=np.roll(u,2,axis=1)
fns={'linear':lambda a,q:a,'cubic':lambda a,q:a**3,'tanh':lambda a,q:np.tanh(q*a),'sin':lambda a,q:np.sin(q*a),'arctan':lambda a,q:np.arctan(q*a),'softsign':lambda a,q:a/(1+q*abs(a)),'rational':lambda a,q:a/(1+q*a*a),'sinh':lambda a,q:np.sinh(q*a)}
def evaluate(own,cross,q,extra=(),detail=False):
 B=np.stack([np.column_stack((np.ones(len(u)),u[:,j],fns[own](u[:,j],q[0]),fns[cross](r[:,j],q[1]),*[x[:,j] for x in extra])) for j in range(4)]); h=0; result=[]; e=[]
 for j,F in enumerate(B):
  Q,R=np.linalg.qr(F); b=np.linalg.lstsq(F,v[:,j],rcond=1e-10)[0];res=F@b-v[:,j]; lev=np.sum(Q**2,axis=1); h+=sum((res/(1-lev))**2);result.append(b);e.append(res)
 return (np.sqrt(h/v.size),np.array(result),np.array(e).T) if detail else np.sqrt(h/v.size)
rows=[]
for ow in fns:
 if ow=='linear':continue
 for cr in fns:
  if cr in ('linear','cubic'):continue
  o=min((minimize(lambda q:evaluate(ow,cr,q),[a,b],bounds=[(.12,5),(.12,5)],method='Nelder-Mead',options={'maxiter':140}) for a,b in [(1,1),(2,2),(3,3)]), key=lambda o:o.fun);rows.append((o.fun,ow,cr,o.x))
rows.sort(key=lambda r:r[0]);print('Top sparse profiles',rows[:12]);
extra={'u^2':u*u,'r^2':r*r,'u*r':u*r,'next':np.roll(u,-1,axis=1),'prev':np.roll(u,1,axis=1),'r':r,'r^3':r**3,'ur^2':u*r*r,'u^2r':u*u*r,'u^5':u**5,'r^5':r**5}
for _,ow,cr,x in rows[:4]:
 score,b,e=evaluate(ow,cr,x,detail=True);print(ow,cr,x,score,'coef',b.round(5),'residual_max',np.max(abs(e)));ex=[]
 for k,t in extra.items():
  o=minimize(lambda q:evaluate(ow,cr,q,(t,)),x,bounds=[(.12,5),(.12,5)],method='Nelder-Mead');ex.append((o.fun,k,o.x))
 print('extra-term comparison',sorted(ex,key=lambda a:a[0])[:6]);
print('x_ranges',[(k,s[k]['x'].min(0),s[k]['x'].max(0)) for k in ['train','test']]);print('telemetry',s['telemetry_vec'])

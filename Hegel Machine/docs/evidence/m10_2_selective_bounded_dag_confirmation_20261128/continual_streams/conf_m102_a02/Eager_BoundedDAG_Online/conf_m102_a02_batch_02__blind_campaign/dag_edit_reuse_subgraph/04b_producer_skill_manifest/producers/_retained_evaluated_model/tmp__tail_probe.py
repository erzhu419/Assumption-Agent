exec(open('/tmp/endpoint_probe.py').read().split('for terms in')[0]); rng=np.random.default_rng(235); I=np.arange(n); tails=[(x[:,j]<=np.quantile(x[:,j],.2) if s<0 else x[:,j]>=np.quantile(x[:,j],.8)) for j in range(x.shape[1]) for s in [-1,1]]; masks=[v for _,v in fs]+tails+[np.isin(I,rng.choice(I,20,replace=False)) for _ in range(30)]
def estimate(m,v,mode):
 a=Q[:,:5]; w=fit(a[m],y[m])[1:]; w/=np.linalg.norm(w)
 def f(i,p):
  z=a[i]@p; z2=z*z; q=Q[i]; aa=a[i]; r=1/(1+z2)
  if mode=='rat': return np.c_[aa,q*r[:,None]]
  if mode=='qr': return np.c_[q,z*r]
  if mode=='ar': return np.c_[q,aa*(1-r)[:,None]]
  if mode=='at': return np.c_[q,aa*np.tanh(z)[:,None]]
  if mode=='tanh': return np.c_[q,np.tanh(z)]
  if mode=='sin': return np.c_[q,np.sin(z)]
  if mode=='erf':
   from scipy.special import erf
   return np.c_[q,erf(z)]
  if mode=='atan': return np.c_[q,np.arctan(z)]
  if mode=='softsign': return np.c_[q,z/np.sqrt(1+z2)]
  if mode=='rpow': return np.c_[q,z*r,z*r*r]
  if mode=='tanhorigin': return np.c_[q,np.tanh(z-a[i,0]*p[0])]
  if mode=='sino': return np.c_[q,np.sin(z-a[i,0]*p[0])]
  if mode=='tanhquad': return np.c_[q,np.tanh(z),np.tanh(z)**2]
  if mode=='sinquad': return np.c_[q,np.sin(z),np.cos(z)]
 def res(p):
  F=f(m,p); return F@fit(F,y[m])-y[m]
 wins=[least_squares(res,np.r_[0.,s*w],bounds=(-5,5),max_nfev=65) for s in (.55,1.25,2.2,3.4)]; best=min(wins,key=lambda o:o.cost); p=best.x; b=fit(f(m,p),y[m]); return f(v,p)@b,p,b
for mode in ['rat','ar','at','qr','tanh','sin','erf','atan','softsign','rpow','tanhorigin','sino','tanhquad','sinquad']:
 full,p,b=estimate(np.ones(n,bool),np.ones(n,bool),mode); errors=np.array([np.mean(abs(estimate(~v,v,mode)[0]-y[v])) for v in masks]); print(mode,'in,public,inter_max,tail_max,random_max:',np.round([np.mean(abs(full-y)),errors[6],max(errors[:6]),max(errors[7:15]),max(errors[15:])],6),'tails',np.round(errors[7:15],4),'p',np.round(p,4),'b',np.round(b,4),flush=True)

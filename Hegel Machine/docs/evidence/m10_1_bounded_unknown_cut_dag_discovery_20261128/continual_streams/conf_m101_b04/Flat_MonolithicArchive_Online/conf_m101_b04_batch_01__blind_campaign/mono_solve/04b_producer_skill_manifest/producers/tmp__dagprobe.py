import itertools as it,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aff(x): return np.c_[np.ones(len(x)),x]
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def poly(x,d): return np.column_stack([np.prod(x[:,p],axis=1) for k in range(d+1) for p in it.combinations_with_replacement(range(x.shape[1]),k)])
def calibrate(s):
 a=s['panel_a']; u,v=a['u'],a['v']; beta=np.array([ls(aff(u[:,[j]]),v[:,j]) for j in range(4)]); r=np.linalg.norm(v-(beta[:,0]+u*beta[:,1]),axis=1); keep=np.argsort(r)[:int(.8*len(r))]; beta=np.array([ls(aff(u[keep][:,[j]]),v[keep,j]) for j in range(4)]); r=np.linalg.norm(v-(beta[:,0]+u*beta[:,1]),axis=1); keep=r<max(.001,8*np.median(r)); beta=np.array([ls(aff(u[keep][:,[j]]),v[keep,j]) for j in range(4)]); return lambda x:beta[:,0]+x*beta[:,1]
def panel_maps(s):
 d=s['panel_c']; u=d['u']; b=s['panel_b']; B=ls(aff(b['u']),b['v']); maps=[]
 def forms(x,p,op,base):
  f=op(x*p[:4]+p[4:]); return np.c_[aff(x),f] if base else aff(f)
 for name,ops,ini in [('v1',[np.tanh,np.sin,lambda x:x/(1+x*x)],1.8),('v2',[np.cos,np.sin],1.0)]:
  v=d[name]; cand=[]
  for op in ops:
   for base in [False,True]:
    def fit(i):
     def fun(p):
      f=forms(u[i],p,op,base); return (f@ls(f,v[i])-v[i]).ravel()
     p=least_squares(fun,np.r_[np.ones(4)*ini,np.ones(4)*.05],bounds=(np.r_[np.ones(4)*.12,np.ones(4)*-2],np.r_[np.ones(4)*5,np.ones(4)*2]),max_nfev=180).x
     return p,ls(forms(u[i],p,op,base),v[i])
    i=np.arange(19); p,w=fit(i); err=np.sqrt(np.mean((forms(u[19:],p,op,base)@w-v[19:])**2)); p,w=fit(np.arange(len(u))); rms=np.sqrt(np.mean((forms(u,p,op,base)@w-v)**2)); print('panel',name,op.__name__,base,'hold',round(err,6),'full',round(rms,6),'p',p.round(3)); cand.append((err,lambda x,p=p,w=w,op=op,base=base:forms(x,p,op,base)@w))
  maps.append(min(cand,key=lambda r:r[0])[1]);
 return (lambda x:aff(x)@B),*maps

def blocks(x,s):
 a=calibrate(s); b,c,d=panel_maps(s); z=a(x); w=b(z); return {'x':x,'a':z,'b':w,'cx':c(x),'dx':d(x),'ca':c(z),'da':d(z),'cb':c(w),'db':d(w),'bca':b(c(z)),'bda':b(d(z))}
if __name__=='__main__':
 x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(x); D=blocks(x,s); Q=poly(x,2); splits=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(13).permutation(n),4)]
 def calc(K,pen):
  es=[]
  for i,j in splits:
   N=np.linalg.qr(Q[i],mode='complete')[0][:,15:]; k=K[np.ix_(i,i)]; w=N@solve(N.T@k@N+pen*np.eye(N.shape[1]),N.T@Y[i],assume_a='pos'); b=ls(Q[i],Y[i]-k@w); e=abs(Q[j]@b+K[np.ix_(j,i)]@w-Y[j]); es.append([e[:,0].mean(),e[:,1:].mean()]);
  return np.array(es)
 Fs={}; Ks={}
 for key,z in D.items():
  for kind,f in [('lin',z),('sq',poly(z,2)[:,1:]),('cube',poly(z,3)[:,1:]),('nonlin',np.c_[np.sin(2*z),np.tanh(2*z),np.cos(2*z),z/(1+z*z)]),('cross',np.einsum('ni,nj->nij',D['a'],z).reshape(n,-1))]:
   f=f-f.mean(0); f=f/np.maximum(f.std(0),1e-8); Fs[key,kind]=f@f.T/f.shape[1];
  zz=z/np.maximum(z.std(0),.15); r=cdist(zz,zz,'sqeuclidean'); Ks[key]=np.exp(-.19*r);
 K0=np.exp(-.23*cdist(x,x,'sqeuclidean'))
 allk={**Fs,**{(k,'rbf'):v for k,v in Ks.items()}}
 allk['mixed','all']=np.mean(list(allk.values()),axis=0)
 allk['mixed','panel']=np.mean([v for (k,t),v in Fs.items() if k in ('ca','da','cb','db')],axis=0)
 rows=[]
 for key,k in allk.items():
  for g in [0,.1,1]:
   for p in [.0001,.001,.01,.1]:
    e=calc(k+g*K0,p); rows.append((e[0,0],e[1:,0].max(),e[0,1],e[1:,1].max(),key,g,p,e[:,0].round(4)))
 for label,rank in [('public',lambda r:r[0]),('worst',lambda r:max(r[0],r[1])),('traj',lambda r:max(r[2],r[3]))]:
  print(label)
  for r in sorted(rows,key=rank)[:10]: print(np.round(r[:4],5),*r[4:])

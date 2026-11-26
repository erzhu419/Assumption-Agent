import sys, pickle, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
from alpha_s00_solver import _aligned_panels, _physical_stresses
D=pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb'))[0]['dataset']; p=D['train']; n=len(p['lab_id']); x,complete=_aligned_panels(D); x=x[:n]; complete=complete[:n]; f=np.c_[np.ones(n),x]; past,hold=_physical_stresses(D,p); dt=D['past_dt']; t=D['eval_times']; tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0]; init=p['initial_memory_states']; y=np.c_[p['strain_curves_obs']-p['strain_0'][:,None],p['instant_rate_obs']]; tt=np.r_[t,1]; labs=p['lab_id']; strata=p['stratum']

def design(l, mode):
 tau=np.exp(l); w=np.exp(-tail[:,None]/tau)*-np.expm1(-dt[:,None]/tau); state=past@w
 if mode!='zero': state+=init*np.exp(-dt.sum()/tau)
 b=np.vstack([-np.expm1(-t[:,None]/tau),1/tau]); a=(hold[:,None]-state)[:,None,:]*b[None]
 z=[a,tt[None,:,None]*f[:,None,:]]
 if mode in ('interact','both'): z.append((a[:,:,:,None]*x[:,None,None,:]).reshape(n,len(tt),-1))
 if mode in ('hold','both'): z.append(tt[None,:,None]*(hold[:,None]*f)[:,None,:])
 return np.concatenate(z,axis=2)

def fit(idx,mode):
 def lin(l):
  a=design(l,mode)[idx]; c=np.linalg.lstsq(a.reshape(-1,a.shape[-1]),y[idx].ravel(),rcond=None)[0]; r=a@c-y[idx]; return np.sum(r*r),c,r
 grid=np.geomspace(.15,16,9); opts=[(lin(l:=np.log([a,b]))[0],l) for i,a in enumerate(grid) for b in grid[i+1:]]; l=min(opts,key=lambda v:v[0])[1]; _,c,r=lin(l)
 def scales(r):
  mad=lambda u: max(1e-5,1.4826*np.median(np.abs(u-np.median(u))))
  return np.r_[np.repeat(mad(r[:,:-1]),len(t)),mad(r[:,-1])]
 for loss in ('soft_l1','cauchy'):
  sc=scales(r); k=len(c)
  p0=np.r_[l,c]; res=least_squares(lambda p: ((design(p[:2],mode)[idx]@p[2:]-y[idx])/sc).ravel(),p0,bounds=(np.r_[[np.log(.051)]*2,[-np.inf]*k],np.r_[[np.log(24.9)]*2,[np.inf]*k]),loss=loss,f_scale=2.5,x_scale='jac',max_nfev=140)
  l,c=res.x[:2],res.x[2:]; r=design(l,mode)[idx]@c-y[idx]
 return np.exp(l),c,design(l,mode)@c-y

if __name__=='__main__':
 for mode in ('zero','base','interact','hold','both'):
  tau,c,r=fit(complete,mode); out=np.zeros_like(y); taus=[]
  for lab in range(4):
   tl,_,rl=fit(complete&(labs!=lab),mode); out[labs==lab]=rl[labs==lab]; taus.append(tl.round(4).tolist())
  def score(v): return np.round([np.sqrt(np.mean(v*v)),np.median(abs(v)),np.quantile(abs(v),.9)],5)
  print(mode,'tau',tau,'theta',c.round(5),'fit',score(r[complete,:-1]),'rate',score(r[complete,-1]),'fold taus',taus)
  print('lab metrics',[(score(out[(labs==l)&complete,:-1]).tolist(),score(out[(labs==l)&complete,-1]).tolist()) for l in range(4)],'missing',score(out[~complete,:-1]),'horizons',np.median(abs(out[complete,:-1]),axis=0).round(4))

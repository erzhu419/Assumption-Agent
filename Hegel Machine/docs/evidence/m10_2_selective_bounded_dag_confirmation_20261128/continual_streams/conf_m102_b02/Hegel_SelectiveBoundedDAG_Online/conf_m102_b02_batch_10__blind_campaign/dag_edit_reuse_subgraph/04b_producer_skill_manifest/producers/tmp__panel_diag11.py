exec(open('/tmp/probe11.py').read().split('A={};')[0])
from itertools import combinations
from scipy.special import erf
u,v=s['panel_a']['u'],s['panel_a']['v']; keep=abs(v)<3
fns={'sin':np.sin,'tanh':np.tanh,'erf':erf,'rat':lambda x:x/(1+x*x),'atan':np.arctan,'soft':lambda x:x/(1+abs(x)),'gauss':lambda x:x*np.exp(-x*x)}
for fn,fun in fns.items():
 for selftype in ['cubic','same','linear']:
  def da(q,j,p):
   a=q[:,j]; b=q[:,(j+2)%4]; return np.c_[np.ones(len(q)),a,*( [a**3] if selftype=='cubic' else [fun(p[1]*a)] if selftype=='same' else []),fun(p[0]*b)]
  def job(p, ret=False):
   cs=[]; rs=[]
   for j in range(4):
    d=da(u,j,p)[keep[:,j]]; y=v[keep[:,j],j]; c=np.linalg.lstsq(d,y,rcond=None)[0]; cs.append(c); rs.extend(d@c-y)
   return (np.array(cs),lambda q:np.column_stack([da(q,j,p)@cs[j] for j in range(4)])) if ret else np.array(rs)
  opts=[least_squares(job,[p,p],bounds=(.05,7),max_nfev=75) for p in [.5,1.2,2.1,3.,4.5]]; op=min(opts,key=lambda o:o.cost); print('A',fn,selftype,np.round(op.x,4),np.std(op.fun),np.round(job(op.x,True)[0],4).tolist())
C=s['panel_c']; u=C['u']; a=C['v1']; b=C['v2']; print('C std/min/max:',[(k,np.std(v,0),v.min(0),v.max(0)) for k,v in C.items() if isinstance(v,np.ndarray) and v.ndim==2]);
for y in [a,b]:
 for z in [u,np.c_[u,u*u],np.c_[u,*[u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]],np.c_[u,u*abs(u)],np.c_[u,u*u,u**3]]:
  d=n._powers(z,1); c=np.linalg.lstsq(d,y,rcond=None)[0]; h=np.einsum('ij,ji->i',d,np.linalg.pinv(d)); print('C',z.shape[1],np.std(d@c-y),np.std((d@c-y)/(1-h[:,None])))
print('B-range',s['panel_b']['u'].min(0),s['panel_b']['u'].max(0)); print('study keys',s.keys(),[(k,list(v)) for k,v in s.items() if isinstance(v,dict)])

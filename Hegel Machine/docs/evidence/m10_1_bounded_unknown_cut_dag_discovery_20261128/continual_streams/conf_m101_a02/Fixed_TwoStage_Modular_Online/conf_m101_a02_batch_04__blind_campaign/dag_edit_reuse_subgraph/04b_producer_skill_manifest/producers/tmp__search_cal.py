import pickle, numpy as np, importlib.util, itertools
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
spec=importlib.util.spec_from_file_location('sink','/holdout_task/_frozen_nodes/node_01.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
fit=lambda X,Y:np.linalg.lstsq(X,Y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def gain(kind,x,k):
 if kind=='rat':return x/(1+k*x*x)
 if kind=='soft':return x/(1+k*abs(x))
 if kind=='tanh':return np.tanh(k*x)
 if kind=='sin':return np.sin(k*x)
 if kind=='cube':return x**3
 if kind=='asinh':return np.arcsinh(k*x)
 if kind=='pow':return np.sign(x)*abs(x)**k
def loo(d,y):
 q=np.linalg.qr(d)[0];return np.linalg.norm(y-q@(q.T@y),axis=1)/(1-(q*q).sum(1))
def make_gain(p,key):
 u,v=p['u'],p[key];out={}; rows=[]
 for kind in ('rat','soft','tanh','sin','asinh','pow','cube'):
  def ds(k,x):return np.c_[aff(x),gain(kind,x,k)]
  opt=minimize_scalar(lambda k: np.mean(loo(ds(k,u),v)**2),bounds=(1.05,5) if kind=='pow' else ((1.6,5) if kind=='sin' else (.05,4)),method='bounded'); k=opt.x
  err=loo(ds(k,u),v); c=fit(ds(k,u),v);rows.append((np.mean(err**2)**.5,max(err),kind,k));out[kind]=lambda x,kind=kind,k=k,c=c: np.c_[aff(x),gain(kind,x,k)]@c
  if kind in ('rat','soft'): print(key,kind,k,'linear and nonlinear',np.round(c,3),flush=True)
 print(key,'gain LOO',sorted(rows),flush=True);return out
B=make_gain(s['panel_b'],'v');C=make_gain(s['panel_c'],'v1'); p=s['panel_c']; C2=lambda x,p=p: np.c_[aff(x),x*x]@fit(np.c_[aff(p['u']),p['u']**2],p['v2'])
p=s['panel_a']; u,v=p['u'],p['v'];A={}
def da(x,mode):
 r=np.roll(x,-1,axis=1); t=x*r
 if mode=='cyc':return np.c_[aff(x),t]
 if mode=='fullq':return np.column_stack([aff(x)]+[x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)])
 if mode=='sinprod':return np.c_[aff(x),np.sin(t)]
 if mode=='prodsin':return np.c_[aff(x),x*np.sin(r)]
 if mode=='tanhprod':return np.c_[aff(x),np.tanh(t)]
 if mode=='aff':return aff(x)
for mode in ('cyc','fullq','sinprod','prodsin','tanhprod','aff'):
 d=da(u,mode);w=np.ones(len(u))
 for _ in range(40):
  c=fit(w[:,None]*d,w[:,None]*v); e=np.linalg.norm(v-d@c,axis=1);w=1/np.hypot(1,e/max(np.median(e),1e-7))
 good=np.argsort(e)[:-3]; print('A',mode,'LOO clean',np.quantile(loo(d[good],v[good]),[.5,.9,1]),flush=True)
 c=fit(d[good],v[good]); A[mode]=lambda x,mode=mode,c=c:da(x,mode)@c
xx=s['train']['x']; n=len(xx); tr=s['train']; results=[]
def inv(f,p):
 u,v=p['u'],p['v']; good=np.linalg.norm(v,axis=1)<4; c=fit(aff(v[good]),u[good]);
 def inverse(x):
  out=[]
  for a,b in zip(x,aff(x)@c):out.append(least_squares(lambda z:f(z[None,:])[0]-a,b,max_nfev=80).x)
  return np.array(out)
 return inverse
ops={'I':lambda x:x,'C':C['soft'],'C2':C2}
for mode in A:ops['A'+mode]=A[mode];ops['a'+mode]=inv(A[mode],s['panel_a'])
for mode in ('rat','soft','tanh','cube','pow'):
 ops['B'+mode]=B[mode];ops['b'+mode]=inv(B[mode],s['panel_b'])
for first in ops:
 if first.startswith('b'):continue
 f=ops[first]; z=f(xx)
 for second in ops:
  if first[0].lower()==second[0].lower() and second!='I':continue
  if second[0].lower()=='a' and first!='I':continue
  if second.startswith('b') and first not in ('I','acyc','afullq','asinprod'):continue
  zz=ops[second](z)
  em=[]
  for va in (np.arange(n-20,n),np.arange(0,n,4)):
   it=np.setdiff1d(np.arange(n),va); st={'eval_times':s['eval_times'],'train':{k:tr[k][it] for k in ('y','traj')}}
   try:
    o=m.run_node({'__input__':st,'node_00':{'z_tr':zz[it],'z_te':zz[va]}});em.append(tuple(np.mean(abs(o[k]-tr[t][va])) for k,t in [('y_hat','y'),('state_pred','traj')]))
   except (ValueError,np.linalg.LinAlgError):break
  if len(em)==2:results.append((max(np.ravel(em)),first,second,em))
print('DAG comparisons:',*sorted(results)[:25],sep='\n')

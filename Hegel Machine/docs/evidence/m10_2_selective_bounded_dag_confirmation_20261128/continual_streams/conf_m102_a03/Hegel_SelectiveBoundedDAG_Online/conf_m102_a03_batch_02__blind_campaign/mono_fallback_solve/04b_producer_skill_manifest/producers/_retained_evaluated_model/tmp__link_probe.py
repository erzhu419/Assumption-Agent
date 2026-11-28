import pickle,runpy,numpy as np
from scipy.optimize import least_squares
S=runpy.run_path('/holdout_task/batch_solver.py'); fit=S['_fit']; A=S['_affine']; Q=S['_quadratic']; s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y,t,x=[tr[k] for k in ('y','traj','x')]; a,b=S['_calibrate'](s['panel_a']); z=(x-a)/b; l=A(z); q=Q(z)
np.set_printoptions(precision=5,suppress=True,linewidth=160)
print('traj sv',np.linalg.svd(t,compute_uv=False)); print('traj linear',fit(l,t)); print('traj quadratic',fit(q,t)[5:]); print('y linear',fit(l,y))
for k in ('panel_b','panel_c'):
 p=s[k]; u=p['u']; print(k)
 for name,v in p.items():
  if name=='u': continue
  coeff=fit(A(u),v); print(name,'affine',np.std(A(u)@coeff-v),coeff)
  for mode in ('tanh','sin','pow2','rat'):
   def B(h):
    return A(np.tanh(u*np.exp(h)) if mode=='tanh' else np.sin(u*np.exp(h)) if mode=='sin' else (u+np.exp(h)*u*u) if mode=='pow2' else u/(1+np.exp(h)*u*u))
   def err(h):
    f=B(h); return (f@fit(f,v)-v).ravel()
   m=min([least_squares(err,np.ones(4)*h,bounds=(-6,2)) for h in (-2,0,1)],key=lambda f:f.cost); f=B(m.x); print(mode,np.sqrt(np.mean(m.fun**2)),m.x)
I=np.arange(52); J=np.arange(52,72); p=np.argsort(np.abs(x@fit(x,y))); rng=np.random.default_rng(512); folds=[(I,J),(p[:52],p[52:])]+[(v[:52],v[52:]) for v in (rng.permutation(72) for _ in range(10))]
for mode in ('fit','state','state2','state+fit','state+fit-reg','state+fit-sin','state+fit-erf'):
 vals=[]
 for i,j in folds:
  st=fit(l[i],t[i]); d=st[1:]@np.ones(t.shape[1]); d/=np.linalg.norm(d); yd=fit(l[i],y[i])[1:]; yd/=np.linalg.norm(yd)
  if d@yd<0: d=-d
  _,_,V=np.linalg.svd(st[1:].T,full_matrices=False); bb=V[:2].T
  qr,_=np.linalg.qr(q[i]); target=y[i]-qr@(qr.T@y[i]);
  def hfull(h):
   if mode=='state': return np.r_[h[0],d*np.exp(h[1])]
   if mode=='state2': return np.r_[h[0],bb@h[1:]]
   return h
  def F(h,k):
   v=l[k]@hfull(h)
   if mode.endswith('sin'): return np.sin(v)[:,None]
   if mode.endswith('erf'):
    from scipy.special import erf
    return erf(v)[:,None]
   return (v/(1+v*v))[:,None]
  def err(h):
   f=F(h,i); g=f-qr@(qr.T@f); e=g@fit(g,target)-target
   if mode.startswith('state+fit'):
    w=h[1:]; e=np.r_[e,(.035 if mode.endswith('reg') else .012)*(w-(w@d)*d)]
   return e
  starts=[np.array([0.,np.log(c)]) if mode=='state' else np.r_[0.,bb.T@yd*c] if mode=='state2' else np.r_[0.,c*yd] for c in (.45,1.2,2.3,3.4)]
  h=min([least_squares(err,h,max_nfev=90) for h in starts],key=lambda a:a.cost).x
  pi=np.c_[q[i],F(h,i)]; pj=np.c_[q[j],F(h,j)]; vals.append(np.mean(abs(pj@fit(pi,y[i])-y[j])))
  if len(vals)<3: print(mode,'split',len(vals),hfull(h))
 print(mode,'cv',np.round(vals[:2],6),'random mean/max',np.mean(vals[2:]),np.max(vals[2:]))

exec(open('/tmp/probe.py').read().split('for d in range(1,6)')[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
Q=poly(x,2); C=poly(x,3)[:,15:]; inds=list(combinations_with_replacement(range(4),3)); fs=[(np.arange(n)%6!=i,np.arange(n)%6==i) for i in range(6)]+[(np.arange(n)<52,np.arange(n)>=52)]
def fit(F,Y): return np.linalg.lstsq(F,Y,rcond=1e-9)[0]
def eval_model(name,model):
 b,e=model(np.arange(n),np.arange(n)); out=[]
 for m,v in fs:
  _,pred=model(np.where(m)[0],np.where(v)[0]); out.append(np.mean(abs(y[v]-pred)))
 print(name,'full',round(np.mean(abs(e-y)),6),'cv',np.round(out,5),'params',np.round(b,4),flush=True)
for terms in [[(0,0,2),(0,2,2),(2,2,2)],[(0,0,2),(0,2,2),(1,2,2),(2,2,2)],[(0,0,2),(0,2,2),(1,2,2),(2,2,2),(2,2,3)]]:
 F=np.c_[Q,C[:,[inds.index(z) for z in terms]]]; eval_model(str(terms),lambda m,v: (fit(F[m],y[m]),F[v]@fit(F[m],y[m])))
def variable(m,v,mode,axial):
 def design(ix,p):
  xs=x[ix]; z=(p[0]*xs[:,2] if axial else xs@p[:4])+p[-1]; a=Q[ix,:5]; q=Q[ix]; r=1/(1+z*z)
  if mode=='q_plus_ar': return np.c_[q,a*(1-r)[:,None]]
  if mode=='a_plus_qr': return np.c_[a,q*r[:,None]]
  if mode=='qr': return q*r[:,None]
  if mode=='q_plus_at': return np.c_[q,a*np.tanh(z)[:,None]]
  if mode=='q_plus_s': return np.c_[q,np.sin(z)]
  if mode=='q_plus_sr': return np.c_[q,z*r]
  if mode=='q_plus_diff': return np.c_[q,np.tanh(z)-z]
  if mode=='q_plus_rankc': return np.c_[q,z**3]
 def resid(p):
  F=design(m,p); return F@fit(F,y[m])-y[m]
 best=None
 for scale in [0.55,1.25,2.2]:
  p=np.array([scale,0.] if axial else [0.,0.,scale,0.,0.]); opt=least_squares(resid,p,bounds=(-4,4),max_nfev=90)
  if best is None or opt.cost<best.cost: best=opt
 p=best.x; F=design(m,p); b=fit(F,y[m]); return p,design(v,p)@b
for mode in ['q_plus_ar','a_plus_qr','qr','q_plus_at','q_plus_s','q_plus_sr','q_plus_diff','q_plus_rankc']:
 for axial in [True,False]:
  if mode=='q_plus_rankc' and axial: continue
  eval_model((mode,axial),lambda m,v: variable(m,v,mode,axial))

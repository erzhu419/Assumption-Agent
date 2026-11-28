import pickle, numpy as np, importlib.util
from scipy.optimize import least_squares
p='/holdout_task/'
sp=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; cal=m._calibrate(s['panel_a']); x=m._inverse(tr['x'],cal); xt=m._inverse(s['test']['x'],cal)
print('range',x.min(0),x.max(0),xt.min(0),xt.max(0)); I=np.arange(len(y)); a0=np.array([-.52,-.15,1.,-.12]); splits=[I]+[np.random.default_rng(k).permutation(I) for k in range(12)]+[np.argsort(g)[::h] for g in [np.linalg.norm(x,axis=1),x@a0] for h in [1,-1]]; print('gate range',np.sort(x@a0)[[0,19,20,51,52,71]],(xt@a0).min(),(xt@a0).max())
A=m._aff(x); Q=m._quadratic(x); L=x[:,[0,1,3]]
for base,gate,reg in [('quad','aff',.012),('quad','aff',.05),('quad','aff',.15),('quad','aff',.4),('quad','fixed',.012),('quad','fixed',.08),('quad','scalar',.002),('diag','aff',.025),('diag','fixed',.025),('tcos','aff',.025),('qsincos','aff',.025),('qsin','aff',.025),('quad','blend',.025),('quad','same',.025)]:
 F={'quad':Q,'diag':np.c_[A,x*x],'tcos':np.c_[A,np.cos(1.25*x),np.cos(2.15*x)],'qsincos':np.c_[Q,np.sin(1.25*x),np.cos(1.25*x)],'qsin':np.c_[Q,np.sin(1.25*x)]}[base]
 def mat(p):
  h=x[:,2]+L@p[:3]+p[3]; k=2.15 if gate=='fixed' else p[4]; v=np.tanh(k*h)
  G=v[:,None]*(1 if gate=='scalar' else A)
  if gate=='blend':G=np.c_[G,np.sin(1.25*h)]
  if gate=='same':G=np.c_[G,np.cos(1.25*h)]
  return np.c_[F,G]
 d=mat([0,0,0,0,2.15]).shape[1]; regularize=np.r_[np.zeros(5),np.full(F.shape[1]-5,reg/2),np.full(d-F.shape[1],reg)]; R=np.diag(regularize); E=[]; delta=[]
 for j,idx in enumerate([I]+splits):
  a,b=(idx,idx) if j==0 else (idx[:52],idx[52:]); Y=np.r_[y[a],np.zeros(d)]
  def calc(p):
   D=np.r_[mat(p)[a],R]; return D,Y,m._ls(D,Y)
  def res(p):
   D,Y,w=calc(p);return D@w-Y
  opts=[least_squares(res,p,bounds=([-3,-3,-3,-2,.035],[3,3,3,2,12]),max_nfev=100) for p in [[-.52,-.15,-.12,0,1.85],[0,0,0,0,2.15],[-.5,0,0,-.5,1.5]]]; r=min(opts,key=lambda r:r.cost); D,_,w=calc(r.x); e=mat(r.x)[b]@w-y[b]; E.append(np.mean(abs(e)))
  if j==0: full=(r.x.round(4),w.round(4),np.std(e)); err=e
 E=np.array(E); print(base,gate,reg,'all',np.round(E[[0,1,-4,-3,-2,-1]],5),'random avg/max',np.round([E[2:14].mean(),E[2:14].max()],5), 'full',full if (base,gate,reg) in [('quad','aff',.012),('quad','fixed',.012),('qsin','aff',.025),('quad','scalar',.002)] else '',flush=True)

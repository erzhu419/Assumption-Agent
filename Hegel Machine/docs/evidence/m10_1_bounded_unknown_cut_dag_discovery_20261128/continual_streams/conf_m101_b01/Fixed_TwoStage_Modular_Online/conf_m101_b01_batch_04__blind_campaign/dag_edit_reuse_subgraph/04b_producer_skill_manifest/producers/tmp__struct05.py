from fit_dag05 import *
from scipy.optimize import least_squares
pc=raw['panel_c']; u=pc['u']; v=pc['v1']

def odd(x,p):
 s=x*x
 return 2.5*x*(1+p[0]*s+p[1]*s*s)/(1+p[2]*s+p[3]*s*s+p[4]*s*s*s)

def panel_res(p):
 A=np.column_stack((np.ones(len(u)),odd(u,p)))
 return (A@np.linalg.lstsq(A,v,rcond=None)[0]-v).ravel()
lo=[0,0,0,0,0]; hi=[100]*5
opt=least_squares(panel_res,[3.66,.01,6.23,2.01,.01],bounds=(lo,hi),max_nfev=400,gtol=1e-11)
print('panel params',repr(opt.x),'rmse',np.sqrt(np.mean(panel_res(opt.x)**2)),flush=True)
ix=np.arange(len(z)); rng=np.random.default_rng(791)
folds=[('official',ix[52:])]+[(f'r{i}',rng.choice(ix,20,replace=False)) for i in range(8)]+[(f'{j}{e}',np.argsort(z[:,j])[:12] if e=='l' else np.argsort(z[:,j])[-12:]) for j in range(4) for e in 'lh']+[('rad',np.argsort(np.linalg.norm(z,axis=1))[-16:])]

def predict(tr,te,p,mode):
 q=odd(z,p); s=z*z; e=s/(1+.164*s)
 extras={'even':e,'paired':z+.542*e,'both':np.column_stack((z,e)),'none':np.empty((len(z),0)),'even_odd':np.column_stack((e,z*e)),'paired_odd':np.column_stack((z+.542*e,z*e))}[mode]
 A=np.column_stack((poly(q,2),extras)); B=A[tr]; Q=np.linalg.qr(B,mode='reduced')[0]; target=y[tr]-Q@(Q.T@y[tr]); x=q[tr]
 def res(p):
  h=np.tanh(x@p[:4]+p[4]); h-=Q@(Q.T@h); c=h@target/(h@h+1e-9)
  return np.r_[target-c*h,3e-5*p,3e-5*c]
 rng=np.random.default_rng(45); starts=[np.zeros(5)+.1]+[rng.normal(0,a,5) for a in (.4,.8,1.5) for _ in range(3)]
 best=min((least_squares(res,w,bounds=(-6,6),max_nfev=140) for w in starts),key=lambda o:o.cost)
 C=np.column_stack((A,np.tanh(q@best.x[:4]+best.x[4])))
 return C[te]@np.linalg.lstsq(C[tr],y[tr],rcond=None)[0],np.linalg.norm(res(best.x))/np.sqrt(len(tr))

for p in (np.array([3.66,0,6.23,2.01,0]),opt.x):
 for mode in ('even','paired','both','none','even_odd','paired_odd'):
  vals=[predict(np.setdiff1d(ix,te),te,p,mode) for _,te in folds]; er=np.array([np.mean(abs(v[0]-y[te])) for v,(_,te) in zip(vals,folds)])
  print('rational',len(p),round(p[1],5),mode,'official/worst/mean',np.round([er[0],max(er),er.mean()],6),folds[er.argmax()][0],'tail',round(er[14],6),'fit rms',round(vals[0][1],6),flush=True)
np.save('/tmp/struct05_panel.npy',opt.x)

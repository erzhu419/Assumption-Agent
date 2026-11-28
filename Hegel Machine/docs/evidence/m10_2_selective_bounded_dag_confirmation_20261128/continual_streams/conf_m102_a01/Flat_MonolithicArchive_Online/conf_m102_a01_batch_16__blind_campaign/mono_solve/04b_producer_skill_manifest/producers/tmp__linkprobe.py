exec(open('/tmp/checkbatch.py').read().split('for j,p in enumerate')[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
P=np.column_stack([np.ones(len(x)),x,*[x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)]])
print('ranges',np.min(y),np.max(y),np.linalg.svd(t,compute_uv=False))
pers=[np.arange(len(x))]+[rng.permutation(len(x)) for _ in range(8)]+[np.argsort(np.linalg.norm(x,axis=1)),np.argsort(y)]
rec=[]
for j,p in enumerate(pers):
 tr,va=p[:52],p[52:]; st={**s,'train':{**r,**{k:r[k][tr] for k in ('x','y','traj')}},'test':{'x':x[va]}}; out=m.solve_batch(st); yy=y[tr]; X=P[tr]; Xv=P[va]; c=np.linalg.lstsq(X,yy,rcond=None)[0]; preds=[]; trainerr=[]
 for name in ('tanh','soft','arctan','sinh','quad_ratio','dual_tanh'):
  if name in ('tanh','soft','arctan','sinh'):
   fun={'tanh':np.tanh,'soft':lambda z:z/(1+abs(z)),'arctan':np.arctan,'sinh':lambda z:np.sinh(np.clip(z,-8,8))}[name]
   def model(A,p):return p[-2]+np.exp(p[-1])*fun(A@p[:-2])
   starts=[np.r_[c/scale,0,np.log(scale)] for scale in (.75,1.5,3,7)]
   def reg(p):return .004*p[:len(c)]
  elif name=='dual_tanh':
   def model(A,p):return A@p[:15]+p[-1]*np.tanh(A[:,:5]@p[15:20])
   starts=[np.r_[c,0,np.random.default_rng(i).normal(size=4)*.8,.1] for i in range(8)]
   def reg(p):return .015*p[5:]
  else:
   def model(A,p):return (A@p[:15])/(1+(A[:,:5]@p[15:])**2)
   starts=[np.r_[c,np.random.default_rng(i).normal(size=5)*.4] for i in range(8)]
   def reg(p):return .012*p[5:]
  opt=min((least_squares(lambda p:np.r_[model(X,p)-yy,reg(p)],p,max_nfev=180) for p in starts),key=lambda q:q.cost); preds.append(model(Xv,opt.x)); trainerr.append(np.mean(abs(model(X,opt.x)-yy)))
 e=lambda v:round(np.mean(abs(v-y[va])),5)
 print(j,e(out['y_hat']),[e(q) for q in preds],[e(.75*out['y_hat']+.25*q) for q in preds],np.round(trainerr,4),flush=True);rec.append((va,out['y_hat'],preds))
pickle.dump(rec,open('/tmp/linkprobe.pkl','wb'))

import sys,importlib.util,numpy as np,pickle
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('bs','/holdout_task/batch_solver.py'); bs=importlib.util.module_from_spec(sp); sp.loader.exec_module(bs)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; ids=np.arange(len(y)); sh,sc=bs._calibrate(s['panel_a']); z=(tr['x']-sh)/sc; P=bs._quad(z)
rng=np.random.default_rng(45); splits=[('gate',ids[:52],ids[52:])]+[(f'r{k}',p[:52],p[52:]) for k in range(8) for p in [rng.permutation(ids)]]
for k,u in enumerate([abs(z[:,2]-.42*z[:,0]),np.linalg.norm(z,axis=1),z[:,2],-z[:,2]]):
 order=np.argsort(u); splits.append((f'e{k}',order[:-12],order[-12:]))

def candidate(i,j,penalty,typ):
 def f(p):
  a,k,d,b,c=p; w=z[:,2]-k*z[:,0]+d+b*z[:,1]+c*z[:,3]; u=a*w*w
  g={'rat':lambda:-u/(1+u),'exp':lambda:np.expm1(-u),'sq':lambda:1/(1+u)**2-1,'root':lambda:1/np.sqrt(1+u)-1}[typ]()
  return np.c_[P,P[:,np.arange(15)!=12]*g[:,None]]
 R=np.diag(np.sqrt(penalty)*np.r_[np.zeros(5),np.ones(10)*.05,np.ones(5)*.1,np.ones(9)])
 def calc(p,ret=False):
  F=f(p); U=np.r_[F[i],R]; Y=np.r_[y[i],np.zeros(len(R))]; c=np.linalg.lstsq(U,Y,rcond=1e-11)[0]; return (F[j]@c,p,c) if ret else U@c-Y
 opt=min([least_squares(calc,p,bounds=([.025,.05,-.8,-.8,-.8],[5,2,.8,.8,.8]),max_nfev=110) for p in [[1,.42,.03,-.1,-.08],[1,.42,-.24,-.1,-.08]]],key=lambda r:r.cost)
 return calc(opt.x,True)
print('public fields',s.keys(),tr.keys(),'test/calibrated max norms',np.linalg.norm((s['test']['x']-sh)/sc,axis=1).max(),np.linalg.norm(z,axis=1).max())
for typ in ['rat','exp','sq','root']:
 for p in [0,.0003,.002,.01,.05,.2]:
  es=[]
  for _,i,j in splits:
   pred,_,_=candidate(i,j,p,typ); es.append(np.mean(abs(pred-y[j])))
  print(typ,p,'gate/mean random/worst random',np.round([es[0],np.mean(es[1:9]),max(es[1:9])],6),'tails',np.round(es[9:],6))
for _,i,j in [splits[0],splits[-3],('all',ids,ids)]:
 pred,p,c=candidate(i,j,0,'rat'); print('rational profile',p.round(4),c.round(4)); print('residual rms',np.sqrt(np.mean((pred-y[j])**2)))

import pickle, numpy as np, importlib.util
from scipy.optimize import minimize_scalar, root
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def load(p):
 sp=importlib.util.spec_from_file_location('mod',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
old=load('/tmp/hegel_continual_archive/dev_m102_c02_batch_08_solver.py'); sink=load('/holdout_task/_frozen_nodes/node_01.py')
A=lambda u:np.c_[np.ones(len(u)),u]; ls=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-10)[0]
def errs(d,y):
 r=y-d@ls(d,y); h=np.sum(d*np.linalg.pinv(d).T,axis=1); return np.mean(abs(r)),np.mean(abs(r/np.maximum(1e-9,1-h[:,None])))
tr=s['train']; x=tr['x']; y=tr['y']; t=tr['traj']; z=old._cal(s['panel_a'],x)
u,v=[s['panel_a'][k] for k in ('u','v')]; w=np.roll(u,-1,axis=1); d=np.c_[A(u),u*w,u**3]; i=np.argsort(np.linalg.norm(v-d@ls(d,v),axis=1))[:-3]; print('A archived residual',np.max(abs(v[i]-d[i]@ls(d[i],v[i]))),'z limits',z.min(0),z.max(0))
print('A local fit',[np.max(abs(v[i,j]-D[i]@ls(D[i],v[i,j]))) for j in range(4) for D in [np.c_[np.ones(len(u)),u[:,j],w[:,j],u[:,j]*w[:,j],u[:,j]**3]]])
def cv(g, split=52):
 st=dict(s,train=dict(x=x[:split],y=y[:split],traj=t[:split])); o=sink.run_node({'__input__':st,'node_00':dict(c=g[:split],query=g[split:])}); return np.mean(abs(o['y_hat']-y[split:])),np.mean(abs(o['state_pred']-t[split:]))
print('z cv',cv(z)); print('archived B cv',cv(old._op(s['panel_b'],'v',z,'b')))
u,v=[s['panel_b'][k] for k in ('u','v')]
fams={'rat':lambda q,a:q/(1+a*q*q),'atan':lambda q,a:np.arctan(a*q),'tanh':lambda q,a:np.tanh(a*q),'sin':lambda q,a:np.sin(a*q),'asinh':lambda q,a:np.arcsinh(a*q),'sqrt':lambda q,a:q/np.sqrt(1+a*q*q),'exp':lambda q,a:q*np.exp(-a*q*q),'radial':lambda q,a:q/(1+a*np.sum(q*q,axis=1,keepdims=True)),'logistic':lambda q,a:q/(1+a*np.abs(q))}
rows=[]
for name,f in fams.items():
 for extra in (0,1,2):
  def D(q,a):
   g=A(f(q,a)); return g if extra==0 else (np.c_[g,q] if extra==1 else np.c_[g,q,q**3])
  def loss(a): return errs(D(u,a),v)[1]
  grid=np.geomspace(.025,12,70); j=np.argmin([loss(a) for a in grid]); a=minimize_scalar(loss,bounds=(grid[max(j-1,0)],grid[min(j+1,len(grid)-1)]),method='bounded').x
  B=D(u,a); g=D(z,a)@ls(B,v); rows.append((loss(a),name,extra,a,errs(B,v)[0],cv(g)))
for r in sorted(rows)[:13]: print(r)
print('affine B',errs(A(u),v),cv(A(z)@ls(A(u),v)))
print('panel B rounded rows:',np.round(np.c_[u,v][:5],5))

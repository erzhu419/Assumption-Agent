import sys,pickle,numpy as np
from itertools import combinations
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; off,gain=calibrate(s); z=(tr['x']-off)/gain; q,a=temporal(s['eval_times'],tr['traj']); T=np.c_[tr['y'],a]; n=52
pairs=list(combinations(range(4),2)); cyc=[(0,1),(0,3),(1,2),(2,3)]
def lin(u): return np.c_[np.ones(len(u)),u]
def cp(u,ps): return np.column_stack([u[:,i]*u[:,j] for i,j in ps])
def c1(u): return np.c_[np.ones(len(u)),cp(u,cyc)]
p=s['panel_c']; b=s['panel_b']; B=np.linalg.lstsq(lin(b['u']),b['v'],rcond=None)[0]; C=np.linalg.lstsq(c1(p['u']),p['v1'],rcond=None)[0]; u=p['u']; v=p['v2']; candidates=[]
for nm,fn,x0,lb,ub in [('rat2',lambda x,k:x*x/(1+k[0]*x*x),[.25],[0],[4]),('rat1',lambda x,k:x*x/(1+k[0]*x+k[1]*x*x),[0,.25],[-.4,0],[.4,4]),('cos',lambda x,k:(1-np.cos(k[0]*x))/k[0]**2,[.6],[.03],[4]),('gauss',lambda x,k:-np.expm1(-k[0]*x*x)/k[0],[.2],[.001],[4]),('tanh',lambda x,k:x*np.tanh(k[0]*x)/k[0],[.6],[.01],[4])]:
 def f(x,k,fn=fn): return np.c_[lin(x),fn(x,k)]
 def err(k):
  F=f(u,k); return (F@np.linalg.lstsq(F,v,rcond=None)[0]-v).ravel()
 opt=least_squares(err,x0,bounds=(lb,ub)); F=f(u,opt.x); W=np.linalg.lstsq(F,v,rcond=None)[0]; print(nm,opt.x,np.std(opt.fun),np.round(W[5:]/W[1:5],3)); candidates.append((np.std(opt.fun),lambda x,f=f,k=opt.x,W=W:f(x,k)@W))
D=min(candidates,key=lambda a:a[0])[1]; ops={'B':lambda x:lin(x)@B,'C':lambda x:c1(x)@C,'D':D}; nodes={'Z':z}; basis={'Z':z};
for _ in range(3):
 for name,x in list(nodes.items()):
  if len(name)!=_+1:continue
  for k,fn in ops.items():
   if k=='B' and name[-1]=='B':continue
   nodes[name+k]=fn(x)
for key,x in nodes.items():
 basis[key]=x; basis[key+'p']=cp(x,pairs); basis[key+'s']=x*x; basis[key+'t']=np.tanh(x); basis[key+'r']=x/(1+.22*x*x)
F0=np.c_[lin(z),z*z,cp(z,pairs)]; base=F0
for stage in range(4):
 scores=[]
 for k,v in basis.items():
  f=np.c_[base,v]; c=np.linalg.lstsq(f[:n],T[:n],rcond=1e-8)[0]; e=f@c-T; scores.append((k,np.mean(abs(e[n:]),0),np.mean(abs(e[:n]),0),np.mean(abs((e[n:,1:])@q.T))))
 for j in range(3): print(stage,j,[(k,np.round(val,4),np.round(train,4),round(h,4)) for k,val,train,h in sorted(scores,key=lambda a:a[1][j])[:6]])
 best=min(scores,key=lambda a:np.sum(a[1])); print('ADDED',best[0]); base=np.c_[base,basis.pop(best[0])]

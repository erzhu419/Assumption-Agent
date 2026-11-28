import pickle
import numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
P=[()] + [j for d in (1,2,3) for j in combinations_with_replacement(range(4),d)]
def design(x,p=P): return np.column_stack([np.prod(x[:,j],axis=1) for j in p])
def ls(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def calibrate(panel):
 u,v=panel['u'],panel['v']; g=design(u); k=np.linalg.norm(v-u,axis=1)<5; w=ls(g[k],v[k]); active=abs(w)>.002
 for j in range(4):
  w[~active[:,j],j]=0; w[active[:,j],j]=ls(g[k][:,active[:,j]],v[k,j]); print('A',j,[(p,round(float(q),6)) for p,q in zip(P,w[:,j]) if q]);
 print('A rmse',np.std(g[k]@w-v[k])); return w
def inverse(x,w):
 u=np.linalg.solve(w[1:5].T,(x-w[0]).T).T
 for _ in range(30):
  r=design(u)@w-x
  if np.max(abs(r))<1e-10: break
  J=np.stack([np.column_stack([j.count(h)*np.prod(u[:,list(j[:j.index(h)]+j[j.index(h)+1:])],axis=1) if h in j else np.zeros(len(u)) for j in P])@w for h in range(4)],axis=2)
  delta=np.linalg.solve(J,r[:,:,None])[:,:,0]; u-=delta/np.maximum(1,np.max(abs(delta),axis=1,keepdims=True))
 print('inversion max error',np.max(abs(design(u)@w-x)),'range',u.min(0),u.max(0));return u
def temporal(t,z):
 def basis(p): return -np.expm1(-t[:,None]*np.exp(p))
 def res(p):
  g=basis(p);return (g@ls(g,z.T)-z.T).ravel()
 return basis(least_squares(res,np.log([.4,1.8])).x)
if __name__=='__main__':
 np.set_printoptions(precision=5,suppress=True); tr=s['train']; x=np.r_[tr['x'],s['test']['x']]; u=inverse(x,calibrate(s['panel_a'])); z=tr['traj']; y=tr['y']; t=s['eval_times']; tb=temporal(t,z[:52]); print('exp amplitude std',ls(tb,z.T).std(1)); print('summary matches',np.array([y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]),tr['summary']);
 b=ls(design(s['panel_b']['u'],P[:5]),s['panel_b']['v']); c=ls(design(s['panel_c']['u'],P[:5]),s['panel_c']['v1']);
 for name,v in [('raw',x),('cal',u),('braw',design(x,P[:5])@b),('bcal',design(u,P[:5])@b),('craw',design(x,P[:5])@c),('ccal',design(u,P[:5])@c)]:
  out=[]
  for q in [v,np.tanh(v),v/np.sqrt(1+.25*v*v),v/(1+.25*v*v),np.sin(v)]:
   for d in [5,15,35]:
    g=design(q,P[:d]); A=g[:52]; coef=ls(A,np.c_[y[:52],ls(tb,z[:52].T).T]); pred=g[52:72]@coef; out.append(tuple(np.round([np.mean(abs(pred[:,0]-y[52:])),np.mean(abs(pred[:,1:]@tb.T-z[52:]))],5)))
  print(name,out)
 for name,v in [('cal',u),('bcal',design(u,P[:5])@b)]:
  for d in [5,15,35]:
   g=design(v[:52],P[:d]); target=np.c_[y[:52],ls(tb,z[:52].T).T]; w=ls(g,target); print(name,d,'training rms',np.std(g@w-target,axis=0));
   if d<=15: print(w)
 np.savez('/tmp/calgraph.npz',u=u,x=x,b=b,c=c)

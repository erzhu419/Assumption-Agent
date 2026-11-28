import sys,pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations_with_replacement as pairs
sys.path.insert(0,'/holdout_task')
from batch_solver import _calibrate,_features
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; off,sc=_calibrate(s['panel_a']); a=(tr['x']-off)/sc; y=tr['y']; z=tr['traj']; t=s['eval_times']; Y=np.c_[y,z]; u=s['panel_c']['u']; v1=s['panel_c']['v1']; v2=s['panel_c']['v2'];
def ls(f,v):return np.linalg.lstsq(f,v,rcond=None)[0]
def lin(a):return np.c_[np.ones(len(a)),a]
def quad(a):return np.column_stack([a[:,i]*a[:,j] for i,j in pairs(range(a.shape[1]),2)])
def basis(a,k,mode):
 h=np.sin(k*a); c=np.cos(k*a)
 return {'s':lambda:np.c_[lin(a),h], 'sc':lambda:np.c_[lin(a),h,c], 'sq':lambda:np.c_[lin(a),h,quad(a)], 'c':lambda:np.c_[lin(a),c], 's3':lambda:np.c_[lin(a),h,np.sin(2*k*a)],'spre':lambda:np.c_[lin(a),np.sin(k*a), a*np.cos(k*a)]}[mode]()
models={}
for num,v,modes,bd in [(1,v1,['s','sc','sq','s3','spre'],(2.7,3.8)),(2,v2,['c'],(1.,1.5))]:
 for m in modes:
  def obj(k):
   f=basis(u,k,m); r=f@ls(f,v)-v; return np.mean(r*r)
  k=minimize_scalar(obj,bounds=bd,method='bounded').x; f=basis(u,k,m); co=ls(f,v); res=v-f@co; lev=np.sum((f@np.linalg.pinv(f))*np.eye(len(u)),axis=1); print('PANEL',num,m,round(k,6),round(np.std(res),6),round(np.std(res/(1-lev[:,None])),6)); models[num,m]=basis(a,k,m)@co
  if m in ['s','c','sq']:
   print('coef',co.round(4).T)
# Identify fixed time shapes by variable projection, without imposing input readout.
for name,fn,x0,bd in [('two exp',lambda k:np.array([-np.expm1(-k[0]*t),-np.expm1(-k[1]*t)]),[1.8,.4],([.8,.02],[4.,1.])),('exp and shifted exp',lambda k:np.array([-np.expm1(-k[0]*t),t*np.exp(-k[1]*t)]),[1.8,.1],([.8,-.8],[4.,1.])),('exp and sin',lambda k:np.array([-np.expm1(-k[0]*t),np.sin(k[1]*t)]),[1.8,.4],([.8,.02],[4.,1.3]))]:
 def r(k):
  f=fn(k);return (z@np.linalg.pinv(f)@f-z).ravel()
 q=least_squares(r,x0,bounds=bd); print('TIME',name,q.x,np.std(q.fun))
rng=np.random.default_rng(312); splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in (rng.permutation(72) for _ in range(12))]; base=_features(a); out=[]
for m in ['s','sc','sq','s3','spre']:
 b,c=models[1,m],models[2,'c']; g=np.c_[b,c];
 for tag,f in [('pair',np.c_[lin(g),.4*quad(g)]),('pair+r',np.c_[lin(g),.4*quad(g),.4*b/(1+b*b),.4*c/(1+c*c)]),('pair+base',np.c_[lin(g),.4*quad(g),.18*base]),('pair+direct',np.c_[lin(g),.4*quad(g),.2*a]),('separate',np.c_[lin(g),.4*quad(b),.4*quad(c)]),('bonly',np.c_[lin(b),.4*quad(b)]),('base',base)]:
  K=f@f.T
  for reg in [.0003,.001,.003,.006,.01]:
   vals=[]
   for i,j in splits:
    yp=K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+reg*np.eye(len(i)),Y[i]); e=abs(yp-Y[j]); vals.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]);
   E=np.array(vals);out.append((m,tag,reg,E[0],E.mean(0),E.max(0)))
for j in [0,1]:
 print('BEST',j)
 for m,tag,reg,e,avg,mx in sorted(out,key=lambda q:q[4][j]+.6*q[5][j])[:13]:print(m,tag,reg,e.round(5),avg.round(5),mx.round(5))

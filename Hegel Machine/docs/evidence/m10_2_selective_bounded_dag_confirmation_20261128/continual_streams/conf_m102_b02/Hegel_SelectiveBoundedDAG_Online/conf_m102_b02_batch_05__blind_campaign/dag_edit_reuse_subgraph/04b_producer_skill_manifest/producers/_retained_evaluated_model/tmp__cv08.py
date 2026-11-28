import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('candidate','/holdout_task/node_00.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=m._invert(s['panel_a'],tr['x']); b=m._panel_b(s['panel_b'],x); c1=m._panel_c(s['panel_c'],b)
p=s['panel_c']; u=p['u']; v=p['v2']
def d2(u,p): return np.c_[np.ones(len(u)),u+u*u/(1+np.exp(p[0])*u*u)*.53]
def r(p):
 d=d2(u,p); return (d@m._fit(d,v)-v).ravel()
a=least_squares(r,[-1.8]).x; c2=d2(b,a)@m._fit(d2(u,a),v)
t=np.c_[tr['y'],tr['traj']]; n=len(t); print('ranges',[(z.min(),z.max()) for z in [x,b,c1,c2,u]]); print('v2 residual',np.sqrt(np.mean(r(a)**2)),a)
for name,c in [('b',b),('c1',c1),('c2',c2),('c1+c2',c1+c2),('c1/c2 as input',m._panel_c(s['panel_c'],c2)),('c1 into c2',d2(c1,a)@m._fit(d2(u,a),v)),('both',np.c_[c1,c2])]:
 d=np.c_[np.ones(n),x,c,(x[:,:,None]*c[:,None,:]).reshape(n,-1)]; vals=[]
 for vl in [np.arange(max(40,n-20),n),*np.array_split(np.arange(n),5)]:
  tl=np.setdiff1d(np.arange(n),vl); pred=d[vl]@m._fit(d[tl],t[tl]); mode=np.linalg.svd(t[tl,1:],full_matrices=False)[2][:2]; pred[:,1:]=(pred[:,1:]@mode.T)@mode; e=np.abs(pred-t[vl]); vals.append([e[:,0].mean(),e[:,1:].mean()]);
 print(name,np.round(vals[0],5),np.round(np.max(vals[1:],axis=0),5))

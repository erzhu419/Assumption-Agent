import runpy,pickle,numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c03_batch_05_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=m['_inverse'](np.vstack([tr['x'],s['test']['x']]),m['_calibration'](s['panel_a'])); one=m['_one']; edge=m['_edge']; fit=m['_fit']; u=s['panel_c']['u']; v=s['panel_c']['v1']; Y=np.column_stack((tr['y'],tr['traj'])); n=len(Y); p=s['panel_b']; b=one(z)@fit(one(p['u']),p['v']); d=one(np.cos(2.4*z))@fit(one(np.cos(2.4*u)),s['panel_c']['v2']);

def err(a,b): return np.sqrt(np.mean((a-b)**2))
def funcs(x,f):
 return {'id':x,'sin':np.sin(f*x),'cos':np.cos(f*x),'tanh':np.tanh(f*x),'edge':edge(x),'sq':x*x,'cube':x**3,'sin+lin':np.c_[np.sin(f*x),x],'tanh+lin':np.c_[np.tanh(f*x),x],'sin+tanh':np.c_[np.sin(f*x),np.tanh(f*x)],'sin+edge':np.c_[np.sin(f*x),edge(x)],'tanh+edge':np.c_[np.tanh(f*x),edge(x)]}
from itertools import combinations
from scipy.optimize import minimize_scalar
pan=[]
for k in funcs(u,1):
 def score(f):
  h=one(funcs(u,f)[k]);return err(h@fit(h,v),v)
 r=minimize_scalar(score,bounds=(.3,3.7),method='bounded'); pan.append((r.fun,k,r.x,score(2.15)))
print('panel bases',sorted(pan))
for k,fun in [('sin',np.sin),('tanh',np.tanh)]:
 rs=[]
 for j in range(4):
  def res(a): return a[0]+a[1]*fun(one(u)@a[2:])-v[:,j]
  w=fit(one(u),v[:,j]); starts=[np.r_[0,1,w],np.r_[0,2,w*.5]]; rs.append(min(err(least_squares(res,a,max_nfev=500).fun,0) for a in starts))
 print('post',k,rs)
A=one(u)@fit(one(p['u']),p['v']); print('v1 first rows',np.round(np.c_[u,A,v],3))
for f in [1.25,1.6,2.15]:
 for k in ['sin','tanh']:
  fn=getattr(np,k); print('linked',f,k,err(one(fn(f*A))@fit(one(fn(f*A)),v),v))

def cols(c,h):
 return np.column_stack([c[:,i]*h[:,j] for i in range(c.shape[1]) for j in range(h.shape[1])])
def cv(h,alpha=0):
 h=h[:n]; out=[]
 for it,iv in [(np.arange(52),np.arange(52,n)),(np.arange(20,n),np.arange(20)),(np.r_[0:26,46:n],np.arange(26,46))]:
  a=h[it]; mean=a.mean(0); sd=np.maximum(a.std(0),1e-8); a=(a-mean)/sd; aa=(h[iv]-mean)/sd; yy=Y[it]-Y[it].mean(0)
  if alpha: w=np.linalg.solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@yy)
  else: w=fit(a,yy)
  e=np.abs(aa@w+Y[it].mean(0)-Y[iv]).mean(0); out.append([e[0],e[1:].mean()])
 return np.round(out,4).tolist()
ca={k:one(funcs(z,f)[k])@fit(one(funcs(u,f)[k]),v) for _,k,f,_ in pan}; ca['b']=b; ca['latent']=z
for k,c in ca.items():
 for t, h in [('dot',np.c_[c,d,np.sum(c*d,1),c[:,0]*d[:,1]]),('full',np.c_[z,edge(z),b,c,d,cols(c,d)]),('poly',np.c_[z,d,cols(z,z),cols(z,d),cols(d,d)])]:
  print(k,t,cv(h,.02))

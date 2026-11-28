import pickle,numpy as np,itertools,importlib.util
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; Y=np.c_[t['y'],t['traj']]
sp=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/conf_m102_b01_batch_06_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m); aff=m._aff;ls=m._ls
z=m._calibrate(s['panel_a'],t['x']); u,v=s['panel_b']['u'],s['panel_b']['v']; p=s['panel_c']; U=p['u'];
forms={'sin+cbrt':(lambda u,q:u+q[0]*np.sin(q[1]*u)+q[2]*np.cbrt(u),[-.4,2.9,0]),'sin+cubic':(lambda u,q:u+q[0]*np.sin(q[1]*u)+q[2]*u**3,[-.4,2.9,0]),'sin+tanh':(lambda u,q:u+q[0]*np.sin(q[1]*u)+q[2]*np.tanh(q[3]*u),[-.4,2.9,0,1.25]),'sin':(lambda u,q:u+q[0]*np.sin(q[1]*u),[-.4,2.9]),'sin+sin':(lambda u,q:u+q[0]*np.sin(q[1]*u)+q[2]*np.sin(q[3]*u),[-.4,2.9,.1,5]),'sin+square':(lambda u,q:u+q[0]*np.sin(q[1]*u)+q[2]*u*u,[-.4,2.9,0]),'sin+sat':(lambda u,q:u+q[0]*np.sin(q[1]*u)+q[2]*u/(1+q[3]*u*u),[-.4,2.9,0,1.25])}
bs={}
for name,(fn,q0) in forms.items():
 def f(x,q):return aff(fn(x,q))
 def res(q):
  a=f(u,q);return (a@ls(a,v)-v).ravel()
 q=least_squares(res,q0,max_nfev=250).x;a=f(u,q); h=(a*np.linalg.pinv(a).T).sum(1); print('B',name,q,np.std(res(q)),np.std(res(q).reshape(-1,4)/(1-h[:,None])));bs[name]=f(z,q)@ls(a,v)
for k,ss in zip(['v2'],[[.35,.065,2.5]]):
 def ev(x,q):return aff(x+q[0]*x*x+q[1]*(1-np.cos(x*q[2])))
 def r(q):
  a=ev(U,q);return (a@ls(a,p[k])-p[k]).ravel()
 q=least_squares(r,ss).x;print('C2',q,np.std(r(q)));D=ev(z,q)@ls(ev(U,q),p[k])
C0=ls(m._pairmap(U),p['v1']);print('C1 coefs',C0);print('B coef',aff(u).T.shape)

def score(f,reg=0.):
 vals=[]
 for a in (52,0,20,40):
  te=np.arange(a,a+20);tr=np.setdiff1d(np.arange(72),te); mu=f[tr].mean(0); std=np.maximum(f[tr].std(0),.0001); F=aff((f-mu)/std); T=F[tr]; d=np.r_[0.,np.full(F.shape[1]-1,reg)]; pred=F[te]@np.linalg.solve(T.T@T+np.diag(d),T.T@Y[tr]) if reg else F[te]@ls(T,Y[tr]); e=abs(pred-Y[te]);vals.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.array(vals)
res=[]
for bname,B in bs.items():
 for source,C in [('B',m._pairmap(B)@C0),('z',m._pairmap(z)@C0)]:
  for ad, dd in [('raw',D),('tanh',np.tanh(1.25*D)),('asinh',np.arcsinh(1.25*D)),('atan',np.arctan(1.25*D))]:
   for ac,c in [('raw',C),('asinh',np.arcsinh(1.25*C)),('tanh',np.tanh(1.25*C)),('atan',np.arctan(1.25*C)),('inv3',np.sinh(np.arcsinh(1.25*C)/3))]:
    P=(c[:,:,None]*dd[:,None,:]).reshape(-1,16)
    for name,f in [('add',np.c_[c,dd]),('cross',np.c_[c,dd,P]),('diag',np.c_[c,dd,c*dd]),('blend',np.c_[c,dd,c*dd,c**2,dd**2]),('crossB',np.c_[c,dd,P,B])]:
     for reg in (.04,.25):
      val=score(f,reg);res.append((val[:,0].mean(),val[:,1].mean(),bname,source,ad,ac,name,reg,np.round(val.T,4)))
for key in [0,1]:
 print('best',key)
 for row in sorted(res,key=lambda r:r[key])[:10]:print(row)

import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(f,v):return np.linalg.lstsq(f,v,rcond=None)[0]
def one(x):return np.ones((len(x),1))
def poly(x,d):return np.column_stack([one(x)]+[np.prod(x[:,i],axis=1) for n in range(1,d+1) for i in cr(range(x.shape[1]),n)])
a=s['panel_a']; u=a['u'];v=a['v']; R=np.empty((2,4))
for j in range(4):
 z=np.array([np.median((v[1:,j]-v[:-1,j])/(u[1:,j]-u[:-1,j])),0]); z[1]=np.median(v[:,j]-u[:,j]*z[0]); keep=np.argsort(abs(v[:,j]-np.c_[u[:,j],one(u)]@z))[:40];R[:,j]=ls(np.c_[u[:,j],one(u)][keep],v[keep,j])
def cal(x):return (x-R[1])/R[0]
tr=s['train']; x=cal(tr['x']); y=tr['y']; t=tr['traj']; b=s['panel_b']; c=s['panel_c']
def fitfun(u,v,fun,z,bounds=(-np.inf,np.inf)):
 def resid(a):
  f=fun(u,a);return (f@ls(f,v)-v).ravel()
 o=least_squares(resid,z,bounds=bounds,max_nfev=250);f=fun(u,o.x); W=ls(f,v)
 return np.linalg.norm(o.fun),o.x,lambda x:fun(x,o.x)@W

def cv(f):
 l=[]
 for p in [np.arange(len(x)),*[np.random.default_rng(j).permutation(len(x)) for j in range(8)]]:
  a,b=p[:52],p[52:];l.append([np.mean(abs(y[b]-f[b]@ls(f[a],y[a]))),np.mean(abs(t[b]-f[b]@ls(f[a],t[a])))])
 return np.round(np.r_[l[0],np.mean(l,axis=0),np.max(l,axis=0)],5)
fs={}
for name,p,v in [('B',b,b['v']),('C1',c,c['v1']),('C2',c,c['v2'])]:
 funcs=[('poly2',lambda u,a:poly(u,2),[0]),('pures3',lambda u,a:np.c_[one(u),u,u*u,u**3],[0]),('pures4',lambda u,a:np.c_[one(u),u,u*u,u**3,u**4],[0]),('sin_cos',lambda u,a:np.c_[one(u),np.sin(a[0]*u),np.cos(a[0]*u)],[2.]),('sin_cos_x',lambda u,a:np.c_[one(u),u,np.sin(a[0]*u),np.cos(a[0]*u)],[2.]),('sincos_sep',lambda u,a:np.c_[one(u),np.sin(u*a),np.cos(u*a)],[2.]*4),('odd_sat',lambda u,a:np.c_[one(u),u/(1+abs(a[0]*u)**a[1])**a[2]],[1.3,1.,1.]),('odd_sat_x',lambda u,a:np.c_[one(u),u,u/(1+abs(a[0]*u)**a[1])**a[2]],[1.3,1.,1.]),('shift_rat',lambda u,a:np.c_[one(u),(u+a[1])/(1+(a[0]*(u+a[1]))**2)],[.4,-1.125]),('shift_rat_x',lambda u,a:np.c_[one(u),u,(u+a[1])/(1+(a[0]*(u+a[1]))**2)],[.4,-1.125]),('rat_x_sq',lambda u,a:np.c_[one(u),u,u*u/(1+(a[0]*u)**2)],[.5])]
 for tag,fun,z in funcs:
  err,a,op=fitfun(p['u'],v,fun,z);f=op(x);fs[name,tag]=f
  print(name,tag,round(err,6),np.round(a,4),'aff',cv(np.c_[one(x),f]),'raw+op',cv(np.c_[one(x),x,f]))
print('pure calib',cv(np.c_[one(x),x,x*x,x**3]));print('pure raw',cv(np.c_[one(x),tr['x'],tr['x']**2,tr['x']**3]))
for tags in [('sin_cos','odd_sat','shift_rat'),('sin_cos_x','odd_sat','shift_rat'),('pures3','odd_sat','shift_rat'),('pures4','odd_sat','shift_rat_x')]:
 f=[fs[j,k] for j,k in zip(('B','C1','C2'),tags)];print('combination',tags,cv(np.c_[one(x),*f]),cv(np.c_[one(x),x,*f])); print('b+c2',cv(np.c_[one(x),f[0],f[2]]),'c1+c2',cv(np.c_[one(x),f[1],f[2]]))

import numpy as np,pickle,itertools
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a=s['panel_a']; u,v=a['u'],a['v']; aff=lambda x:np.c_[np.ones(len(x)),x]; ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]; ok=np.linalg.norm(v-aff(u)@ls(aff(u),v),axis=1)<4; u,v=u[ok],v[ok]
rot=np.roll(np.arange(4),2); w=[]
for j in range(4):
 def af(x,k): return np.c_[aff(x),x[:,[j,rot[j]]]**3,np.sin(k*x[:,[j,rot[j]]])]
 def fun(k):
  f=af(u,k); return np.linalg.norm(f@ls(f,v[:,j])-v[:,j])
 opt=minimize_scalar(fun,bounds=(2,3.6),method='bounded'); w.append((opt.x,ls(af(u,opt.x),v[:,j]),j)); print('A',j,opt.x,opt.fun,w[-1][1]);
def ac(x):
 return np.column_stack([np.c_[aff(x),x[:,[j,rot[j]]]**3,np.sin(k*x[:,[j,rot[j]]])]@b for k,b,j in w])
xraw=np.vstack([s['train']['x'],s['test']['x']]); x=[]
for row in xraw:
 x.append(least_squares(lambda z:ac(z[None])[0]-row,(row-np.array([.09,-.08,-.05,.02]))/1.2,bounds=(-2.5,2.5)).x)
x=np.array(x); print('invert',np.max(abs(ac(x)-xraw)),x.min(0),x.max(0))
B=s['panel_b']; C=s['panel_c']; zz={}; zz['u']=x; zz['raw']=xraw
fb=lambda u:aff(np.cos(2.6*u)); zz['b']=fb(x)@ls(fb(B['u']),B['v']);
fc2=lambda u:np.c_[aff(u),u*u]; zz['c2']=fc2(x)@ls(fc2(C['u']),C['v2']);
for mode in ['tan','rat','tanh','sinh','cubic','sin']:
 def t(u,k):return {'tan':np.arctan(k*u),'rat':u/np.sqrt(1+k*u*u),'tanh':np.tanh(k*u),'sinh':np.arcsinh(k*u),'cubic':u+k*u**3,'sin':np.sin(k*u)}[mode]
 for aug in ['none','lin','sq','linsq']:
  def f(u,k):return np.c_[aff(t(u,k)),*[z for flag,z in [('lin',u),('sq',u*u)] if flag in aug]]
  def obj(k):
   g=f(C['u'],k); return np.linalg.norm(g@ls(g,C['v1'])-C['v1'])
  r=minimize_scalar(obj,bounds=(.1,7),method='bounded'); name=mode+'_'+aug; zz[name]=f(x,r.x)@ls(f(C['u'],r.x),C['v1']); print('C',name,round(r.x,5),round(r.fun,6))
Y=np.c_[s['train']['y'],s['train']['traj']]; rr=np.random.default_rng(9); inds=[np.arange(72),*[rr.permutation(72) for _ in range(4)]]
def cv(f,l):
 out=[]
 for i in inds:
  tr,te=i[:52],i[52:]; g=f[tr]; p=f[te]@np.linalg.solve(g.T@g+l*np.eye(g.shape[1]),g.T@Y[tr]); e=abs(p-Y[te]);out.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.array(out)
def poly(u,d):return np.column_stack([aff(u)]+[np.prod(u[:,j],axis=1) for k in range(2,d+1) for j in itertools.combinations_with_replacement(range(u.shape[1]),k)])
opts=[]
for nm in zz:
 for d in [1,2,3]:
  for extr in ['', 'u','b','c2','u_b','b_c2','u_b_c2']:
   f=np.c_[poly(zz[nm],d),*[zz[j] for j in extr.split('_') if j and j!=nm]]
   for l in [.0001,.001,.01,.1]:
    e=cv(f,l); opts.append((e.max(),nm,d,extr,l,e[0],e.mean(0)))
for row in sorted(opts,key=lambda z:z[0])[:20]:print('CV',row)
np.savez('/tmp/features11.npz',**zz)

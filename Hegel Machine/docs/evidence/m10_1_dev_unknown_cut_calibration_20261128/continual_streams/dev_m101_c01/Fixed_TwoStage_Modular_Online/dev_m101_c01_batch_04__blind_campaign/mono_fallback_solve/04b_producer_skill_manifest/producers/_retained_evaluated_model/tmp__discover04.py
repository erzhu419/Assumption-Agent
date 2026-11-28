from cfit04 import *
from scipy.optimize import minimize_scalar

def ls(h,v): return np.linalg.lstsq(h,v,rcond=None)[0]
def projected(u,v,feature,init,bounds,linear=True):
 def h(x,p): return np.c_[aff(x) if linear else np.ones(len(x)),feature(x,p)]
 def obj(p):
  a=h(u,p); return (a@ls(a,v)-v).ravel()
 fit=least_squares(obj,init,bounds=bounds,max_nfev=200); w=ls(h(u,fit.x),v)
 return lambda x:h(x,fit.x)@w,fit.x

laws={
 'atan':(lambda x,p: np.arctan(p[0]*x),[2],([.1],[8])),
 'softpower':(lambda x,p:x/(1+p[0]*x*x)**p[1],[2,.5],([.01,.05],[25,6])),
 'ratquart':(lambda x,p:x/(1+p[0]*x*x+p[1]*x**4),[2,.1],([.01,0],[25,25])),
 'ratabs':(lambda x,p:x/(1+p[0]*abs(x)**p[1]),[2,2],([.01,.25],[25,8])),
 'softabs':(lambda x,p:x/(1+p[0]*abs(x))**p[1],[2,1],([.01,.05],[25,6])),
 'erf':(lambda x,p:__import__('scipy').special.erf(p[0]*x),[2],([.1],[8])),
 'mix':(lambda x,p:np.tanh(p[0]*x)+p[2]*np.sin(p[1]*x),[2,1,.1],([.1,.1,-5],[8,8,5])),
 'sqrat':(lambda x,p:x*x/(1+p[0]*x*x),[.2],([.001],[8])),
 'sqroot':(lambda x,p:np.sqrt(1+p[0]*x*x)-1,[.2],([.001],[8]))}

def fit_all(s):
 P=s['panel_a']; u0=P['u']; v=P['v']; keep=np.ones(len(v),bool)
 for _ in range(5):
  aa=np.array([ls(aff(u0[keep,j:j+1]),v[keep,j]) for j in range(4)]).T; e=np.linalg.norm(aa[0]+u0*aa[1]-v,axis=1); keep=e<max(.001,4*np.median(e))
 P=s['panel_b']; bb=ls(aff(P['u']),P['v']); print('B affine rms',np.sqrt(np.mean((aff(P['u'])@bb-P['v'])**2)))
 P=s['panel_c']; u=P['u']; out=[]
 for key,keys in [('v1',list(laws)[:-2]),('v2',list(laws)[-2:]+['cos'])]:
  v=P[key]; choices=[]
  for name in keys:
   args=laws[name] if name!='cos' else (lambda x,p:np.cos(p[0]*x)-1,[1],([.02],[5]))
   fc,p=projected(u[:18],v[:18],*args); err=np.sqrt(np.mean((fc(u[18:])-v[18:])**2)); choices.append((err,name,args)); print('C',key,name,round(err,7),p.round(5))
  out.append(projected(u,v,*min(choices)[2])[0])
 return aa,bb,*out

if __name__=='__main__':
 A,B,C,D=fit_all(s); x=s['train']['x']; z=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; target=np.c_[y,z]; random=np.random.default_rng(91)
 cv=[(np.arange(56),np.arange(56,72))]+[(i[:56],i[56:]) for i in (random.permutation(72) for _ in range(4))]
 paths={'x':x,'a':A[0]+x*A[1],'i':(x-A[0])/A[1]}; source=list(paths.items());
 for k,u in source: paths[k+'B']=aff(u)@B
 for k,u in list(paths.items()):
  paths[k+'C']=C(u);paths[k+'D']=D(u);paths[k+'CD']=np.c_[C(u),D(u)]
 rows=[]
 for k,u in paths.items():
  fs={'L':aff(u),'Q':poly(u,2),'Qx':np.c_[poly(x,2),u],'LxQ':np.c_[poly(x,2),u,u*u]}
  if u.shape[1]==4:fs.update({'Qc':np.c_[poly(u,2),poly(x,2)[:,1:]],'mix':np.c_[poly(x,2),u,(u[:,:,None]*x[:,None,:]).reshape(len(x),-1)]})
  for n,H in fs.items():
   for pen in ([0,.03,.12] if n!='L' else [0]):
    e=[]
    for i,j in cv:
     p=H[j]@ls(np.r_[H[i],pen*np.eye(H.shape[1])],np.r_[target[i],np.zeros((H.shape[1],9))]); d=abs(p-target[j]); e.append([d[:,0].mean(),d[:,1:].mean()])
    rows.append((k,n,pen,np.array(e)))
 for col in [0,1]:
  print('BEST',col)
  for k,n,p,e in sorted(rows,key=lambda r:r[3][:,col].mean())[:14]: print(k,n,p,e[:,col].round(5),'mean',e[:,col].mean().round(5))

import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import product,combinations
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];T=s['train']['traj'];y=s['train']['y'];t=s['eval_times'];np.set_printoptions(precision=5,suppress=True,linewidth=150)
def aff(p,key='v'):
 u=p['u'];v=p[key];f=np.c_[np.ones(len(u)),u];i=np.arange(len(u))
 for _ in range(6):
  b=np.linalg.lstsq(f[i],v[i],rcond=None)[0];i=np.argsort(np.linalg.norm(f@b-v,axis=1))[:int(.85*len(u))]
 return b[0],b[1:]
a,A=aff(s['panel_a']);b,B=aff(s['panel_b']);c,C=aff(s['panel_c'],'v1');d,D=aff(s['panel_c'],'v2');print(a,A,b,B,sep='\n'); print('sv traj',np.linalg.svd(T,compute_uv=False));
def hat(F,Y):return np.linalg.lstsq(F,Y,rcond=None)[0]
def basis(p):return -np.expm1(-t[:,None]*p[None,:])
def err(p):
 f=basis(p);return (f@hat(f,T.T)-T.T).ravel()
r=least_squares(err,[.5,2],bounds=(.001,15));F=basis(r.x); Z=hat(F,T.T).T;print('time rates',r.x,'rmse',np.std(err(r.x)),'basis',F,'coef',Z[:8]);print('other time fits');
for fam in [lambda p:np.c_[t,-np.expm1(-p[0]*t)],lambda p:np.c_[-np.expm1(-p[0]*t),t*np.exp(-p[0]*t)],lambda p:np.c_[-np.expm1(-p[0]*t),-np.expm1(-p[0]*t)**2]]:
 r0=least_squares(lambda p:(fam(p)@hat(fam(p),T.T)-T.T).ravel(),[1.]);print(r0.x,np.std(r0.fun))
for k in ['panel_c']:
 p=s[k];
 for key in ['v1','v2']:
  for deg in [1,2,3]:
   f=PolynomialFeatures(deg).fit_transform(p['u']); print(key,deg,np.std(p[key][18:]-f[18:]@hat(f[:18],p[key][:18])));
frames={'raw':x,'Aforward':x@A+a,'Ainverse':(x-a)@np.linalg.inv(A)}
for k,v in list(frames.items()):
 frames[k+'/B']=v@B+b;frames[k+'/Binv']=(v-b)@np.linalg.inv(B);frames[k+'/C']=v@C+c;frames[k+'/D']=v@D+d
for k,v in frames.items():
 for idx, h in enumerate(np.c_[Z,y].T):
  f=PolynomialFeatures(3).fit_transform(v);f1=f[:,:15]; beta=hat(f1,h); ev=np.mean(abs(h[52:]-f1[52:]@hat(f1[:52],h[:52]))); print(k,idx,round(ev,4),'coef',beta.round(3)) if k in ('raw','Ainverse','Ainverse/B','Ainverse/Binv') else None
  feats=[np.ones(len(v))];names=['1'];
  for j,name in enumerate('abcd'):
   z=v[:,j];
   for nm,g in [('x',z),('x2',z*z),('x3',z**3),('tanh',np.tanh(z)),('sin',np.sin(z)),('cos',np.cos(z)),('atan',np.arctan(z)),('sat',z/(1+z*z))]:feats.append(g);names.append(name+nm)
  for j,l in combinations(range(4),2):
   q,w=v[:,j],v[:,l];
   for nm,g in [('mul',q*w),('div',q/(1+w*w)),('idiv',w/(1+q*q)),('sin+',np.sin(q+w)),('sin-',np.sin(q-w)),('tanh+',np.tanh(q+w)),('tanh-',np.tanh(q-w))]:feats.append(g);names.append('abcd'[j]+'abcd'[l]+nm)
  for j,l,m in combinations(range(4),3):feats.append(v[:,j]*v[:,l]*v[:,m]);names.append('abcd'[j]+'abcd'[l]+'abcd'[m]);
  f=np.array(feats).T;sel=[0];res=h[:52]-h[:52].mean()
  for step in range(10):
   fr=f[:52]-f[:52,sel]@hat(f[:52,sel],f[:52]); wow=np.abs(fr.T@res)/np.maximum(1e-9,np.linalg.norm(fr,axis=0));wow[sel]=0;sel.append(np.argmax(wow));bb=hat(f[:52,sel],h[:52]);res=h[:52]-f[:52,sel]@bb
   test=np.mean(abs(h[52:]-f[52:,sel]@bb));
   if step+1 in (4,7,10):print('sparse',k,idx,step+1,round(np.mean(abs(res)),3),round(test,3),','.join(names[i] for i in sel))

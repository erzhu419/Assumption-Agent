import numpy as np,pickle
from itertools import combinations_with_replacement as cr
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(x)
def des(z,d):return np.column_stack([np.ones(len(z))]+[np.prod(z[:,ix],axis=1) for k in range(1,d+1) for ix in cr(range(4),k)])
def fit(a,b,keep=None):
 if keep is None:return np.linalg.lstsq(a,b,rcond=None)[0]
 return np.linalg.lstsq(a[keep],b[keep],rcond=None)[0]
p=s['panel_a'];u,v=p['u'],p['v']; h=des(u,1);keep=np.linalg.norm(v-h@fit(h,v),axis=1)<2
for d in (1,2,3):
 h=des(u,d);b=fit(h,v,keep);print('A degree',d,'errors',np.quantile(np.linalg.norm((v-h@b)[keep],axis=1),[0,.5,1]));
 if d==2: print(b.round(4)); ba=b
cal=np.linalg.lstsq(des(u,1)[keep],v[keep],rcond=None)[0]; dd=np.array([fit(des(u,1)[:,[0,j+1]],v[:,j],keep) for j in range(4)])
def inv(x):
 return np.array([least_squares(lambda z:des(z[None,:],2)[0]@ba-row,(row-cal[0])@np.linalg.inv(cal[1:])).x for row in x])
coords={'raw':x,'diag':(x-dd[:,0])/dd[:,1],'aff':(x-cal[0])@np.linalg.inv(cal[1:]),'qua':inv(x)}
p=s['panel_b']; B=fit(des(p['u'],1),p['v']);
p=s['panel_c']; u=p['u'];
for k in ('v1','v2'):
 v=p[k];bc=fit(des(u,1),v); print(k,'bias',bc[0]);
 for kind,fn in [('gauss',lambda z,a:np.expm1(-a*z*z)),('tanh',lambda z,a:np.tanh(a*z)),('sin',lambda z,a:np.sin(a*z)),('cos',lambda z,a:np.cos(a*z))]:
  def hf(z,a):return np.c_[des(z,1),fn(z,a)]
  def er(a):
   h=hf(u,a);return (v-h@fit(h,v)).ravel()
  a=least_squares(er,[.6],bounds=([.03],[5])).x[0]; h=hf(u,a); w=fit(h,v);print(kind,a,np.std(er(a)))
  if kind=='gauss':
   print('coef',w.round(4),'ratio', (w[5:]@np.linalg.inv(w[1:5])).round(4));
   for key,z in list(coords.items())[:4]:
    for mul in (.5,1.,1.25,2.):
     coords[(key,k,mul)]=z+mul*fn(z,a)@w[5:]@np.linalg.inv(w[1:5]);
for key,z in list(coords.items()):
 if isinstance(key,str):coords[key+'B']=z@B[1:]/.88
rng=np.random.default_rng(1); splits=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(rng.permutation(n),4)]
def score(f,reg):
 ans=[]
 for i,j in splits:
  m=f[i].mean(0); sd=np.maximum(f[i].std(0),1e-8); A=(f[i]-m)/sd;D=(f[j]-m)/sd; ym=Y[i].mean(0); E=D@A.T@np.linalg.solve(A@A.T+reg*np.eye(len(i)),Y[i]-ym)+ym-Y[j];ans.append([np.mean(abs(E[:,0])),np.mean(abs(E[:,1:]))]);
 return np.asarray(ans)
R=[]
for key,z in coords.items():
 for mode, f in [('q',des(z,2)),('c',des(z,3)),('qt',np.c_[des(z,2),des(np.tanh(z),2)]),('ct',np.c_[des(z,3),des(np.tanh(z),2)]),('sin',np.c_[des(z,2),np.sin(z),np.cos(z),np.sin(2*z),np.cos(2*z)])]:
  for reg in (.003,.03,.3):
   sc=score(f,reg);R.append((max(sc[0]),key,mode,reg,sc[0],sc.max(0),sc.mean(0)))
for r in sorted(R,key=lambda r:r[0])[:20]:print(r)
for k in [4,5]:
 print('best',k);print(*sorted(R,key=lambda r:max(r[k]))[:5],sep='\n')

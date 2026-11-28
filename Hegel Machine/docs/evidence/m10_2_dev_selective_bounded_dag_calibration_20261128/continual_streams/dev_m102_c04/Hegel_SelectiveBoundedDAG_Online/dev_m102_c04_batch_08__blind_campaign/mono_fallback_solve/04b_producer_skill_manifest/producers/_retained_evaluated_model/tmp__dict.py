import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
pa=s['panel_a']; u,v=pa['u'],pa['v']; m=np.linalg.norm(u-v,axis=1)<1; w=np.array([ls(aff(u[m])[:,[0,j+1]],v[m,j]) for j in range(4)]).T
pb=s['panel_b']; wb=ls(aff(pb['u']),pb['v']); B=lambda x:aff(x)@wb
pairs=[(0,1),(0,3),(1,2),(2,3)]; prod=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in pairs]); cos=lambda x:np.cos(2.4*x)
pc=s['panel_c']; wc1=ls(aff(prod(pc['u'])),pc['v1']); wc2=ls(aff(cos(pc['u'])),pc['v2']); C=lambda x:aff(prod(x))@wc1; D=lambda x:aff(cos(x))@wc2
x=s['train']['x']; z=x*w[1]+w[0]; Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; tr=Y[:,1:];
def temporal(p):
 return -np.expm1(-t[:,None]*p)
o=least_squares(lambda p:(temporal(p)@ls(temporal(p),tr[:52].T)-tr[:52].T).ravel(),[.5,2.5],bounds=(.01,30)); print('best rates',o.x); print('B',wb,'C',wc1,'D',wc2,sep='\n');
base={'x':x,'z':z,'inv':(x-w[0])/w[1]}; sets=dict(base)
for nm,a in base.items():
 for name,op in [('B',B),('C',C),('D',D)]:sets[name+nm]=op(a)
for nm,a in list(sets.items()):
 if len(nm)==2 and nm[0] in 'BCD':
  for name,op in [('B',B),('C',C),('D',D)]:sets[name+nm]=op(a)

from scipy.spatial.distance import cdist
from scipy.linalg import qr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import MultiTaskLasso
np.set_printoptions(precision=4,suppress=True)
print('nearest data'); ds=cdist(x,x); np.fill_diagonal(ds,np.inf); nn=ds.argmin(1); ii=np.argsort(ds.min(1))[:15]; print(np.c_[ii,nn[ii],ds[ii,nn[ii]],Y[ii,0],Y[nn[ii],0],Y[ii,-1],Y[nn[ii],-1]])
F=[]; names=[]
def add(name,h):
 h=np.asarray(h).reshape(len(x),-1); F.append(h); names.extend(f'{name}:{i}' for i in range(h.shape[1]))
# Include transformed linear coordinates with and without operator bias, and inverse coordinates.
for k in ['x','z','inv']:
 sets['BI'+k]=(sets[k]-wb[0])@np.linalg.inv(wb[1:]); sets['B0'+k]=sets[k]@wb[1:]; sets['BT'+k]=sets[k]@wb[1:].T+wb[0]
for k,h in list(sets.items()):
 if k.startswith(('BI','BT','B0')): sets['C'+k]=C(h); sets['D'+k]=D(h)
for k,h in sets.items():
 add(k,h)
 for a in [.22,.5,1.,1.25,1.8,2.4,3.5,5.]:
  for n,f in [('tanh',np.tanh(a*h)),('sin',np.sin(a*h)),('cos',np.cos(a*h)),('r1',h/(1+a*h*h)),('r2',h*h/(1+a*h*h)),('soft',h/(1+a*abs(h)))]:add(f'{n}{a}({k})',f)
 add('abs'+k,abs(h));add('norm'+k,np.sum(h*h,1));add('prod'+k,PolynomialFeatures(3,include_bias=False).fit_transform(h)[:,4:])
for (k,h),(l,g) in it.combinations(sets.items(),2):
 if k in ['x','z','Bz','Bx','Binv','Cz','Dz'] or l in ['x','z','Bz','Bx','Binv','Cz','Dz']:
  add(f'{k}*{l}',(h[:,:,None]*g[:,None,:]).reshape(-1,16)); add(f'dot{k}*{l}',np.sum(h*g,1)); add(f'sqdist{k}-{l}',np.sum((h-g)**2,1));
  for a in [.22,1.25]:
   add(f'ratio{k}/{l}/{a}',h/(1+a*g*g));add(f'ratio{l}/{k}/{a}',g/(1+a*h*h));add(f'rg{k}/{l}/{a}',h/(1+a*np.sum(g*g,1,keepdims=True)));add(f'rg{l}/{k}/{a}',g/(1+a*np.sum(h*h,1,keepdims=True)))
F=np.hstack(F); F-=F[:52].mean(0); F/=np.maximum(F[:52].std(0),1e-9); names=np.asarray(names); yy=Y-Y[:52].mean(0); U,S,V=np.linalg.svd(tr[:52],full_matrices=False); T=np.c_[yy[:,0],(tr-tr[:52].mean(0))@V[:3].T/S[:3]*np.sqrt(52)];
for ti in range(5):
 target= T[:,ti:ti+1] if ti<4 else T
 r=target[:52].copy(); h=F[:52].copy(); ix=[]
 for k in range(20):
  scr=np.sum((h.T@r)**2,axis=1)/np.maximum(np.sum(h*h,axis=0),1e-8); j=np.argmax(scr); v=h[:,j]/np.linalg.norm(h[:,j]); h-=v[:,None]*(v@h); r-=v[:,None]*(v@r); ix.append(j)
  if k<6 or k in [9,14,19]:
   zz=F[:,ix]; p=zz@ls(zz[:52],target[:52]); print(ti,k+1,names[j],np.mean(abs(r)),np.mean(abs(p[52:]-target[52:])))
for alpha in [.003,.01,.03,.06,.1,.2]:
 m=MultiTaskLasso(alpha=alpha,fit_intercept=False,max_iter=2000).fit(F[:52],T[:52]); e=abs(m.predict(F[52:])-T[52:]).mean(0); jj=np.argsort(np.linalg.norm(m.coef_,axis=0))[-8:][::-1];print('lasso',alpha,e,names[jj])

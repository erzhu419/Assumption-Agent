import pickle,itertools as it,numpy as np
from scipy.optimize import least_squares
from scipy.linalg import svd
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; np.set_printoptions(precision=4,suppress=True,linewidth=145)
pa=s['panel_a']; ua,va=pa['u'],pa['v']; a=np.array([np.median((va[i,j]-va[k,j])/(ua[i,j]-ua[k,j])) for j in range(4) for i,k in []]); a=[]; b=[]
for j in range(4):
 i,k=np.triu_indices(len(ua),1); m=np.median((va[i,j]-va[k,j])/(ua[i,j]-ua[k,j])); c=np.median(va[:,j]-m*ua[:,j]); keep=abs(va[:,j]-m*ua[:,j]-c)<.02; c,m=np.linalg.lstsq(np.c_[np.ones(sum(keep)),ua[keep,j]],va[keep,j],rcond=None)[0]; a.append(m);b.append(c)
a=np.array(a); b=np.array(b); print('A:',a,b)
pb=s['panel_b']; B=np.linalg.lstsq(np.c_[pb['u'],np.ones(len(pb['u']))],pb['v'],rcond=None)[0]; print('B singulars',svd(B[:4])[1]);
coords={'raw':x,'Ainv':(x-b)/a,'B':np.c_[x,np.ones(len(x))]@B,'AinvB':np.c_[(x-b)/a,np.ones(len(x))]@B,'AinvBinv':((x-b)/a-B[-1])@np.linalg.inv(B[:4])};
U,D,V=svd(tr,full_matrices=False); z=U[:,:2]*D[:2]; print('t',t,'temporal V',V[:2]);
for label,func,n,p in [('two exp',lambda p:np.c_[1-np.exp(-p[0]*t),1-np.exp(-p[1]*t)],2,[.4,2]),('exp+linear',lambda p:np.c_[1-np.exp(-p[0]*t),t],1,[1]),('exp+delayed',lambda p:np.c_[1-np.exp(-p[0]*t),t*t/(1+p[1]*t)],2,[1,1]),('forced shared decay',lambda p:np.c_[1-np.exp(-p[0]*t),(np.exp(-p[1]*t)-np.exp(-p[0]*t))/(p[0]-p[1])],2,[2,.2]),('exp+resonance',lambda p:np.c_[1-np.exp(-p[0]*t),t*np.exp(-p[0]*t)],1,[1])]:
 f=lambda p:((V[:2]@np.linalg.pinv(func(p).T))@func(p).T-V[:2]).ravel(); sol=least_squares(f,p,bounds=(1e-6,30)); print(label,sol.x,np.linalg.norm(sol.fun));
 if label=='two exp': z2=tr@np.linalg.pinv(func(sol.x).T)
pc=s['panel_c']; u=pc['u'];
for name in ['v1','v2']:
 v=pc[name]; print('C',name)
 for fname,fn,inv in [('aff',lambda p:p,lambda p:p),('tanh',np.tanh,lambda p:np.arctanh(np.clip(p,-.999,.999))),('sin',np.sin,lambda p:np.arcsin(np.clip(p,-1,1))),('sinh',np.sinh,np.arcsinh),('asinh',np.arcsinh,np.sinh),('cubic',lambda p:p+.2*p**3,lambda p:p)]:
  F=np.c_[np.ones(len(u)),u]; start=np.linalg.lstsq(F,inv(v),rcond=None)[0]; models=[]; errs=[]
  for j in range(4):
   f=lambda q: q[5]*fn(F@q[:5])+q[6]-v[:,j]; q=least_squares(f,np.r_[start[:,j],1.2,0],max_nfev=300).x; errs.append(np.mean(abs(f(q)))); models.append(q)
  print(fname,errs,np.array(models).round(3) if max(errs)<.01 else '')

def features(q):
 fs=[]; names=[]
 def add(name,a):names.append(name);fs.append(a)
 for k in range(1,5):
  for ii in it.combinations_with_replacement(range(4),k):add(''.join(map(str,ii)),np.prod(q[:,ii],axis=1))
 for i in range(4):
  for j in range(i+1,4):
   for k in [-1,1]:add(f'ratio{i}{j}{k}',q[:,i]/(1+(q[:,i]+k*q[:,j])**2))
 for i in range(4):
  for w in [.5,1,1.5,2,2.5,3]:
   for name,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rational',lambda p:p/(1+p*p))]:add(f'{name}({w}*{i})',fn(w*q[:,i]))
 for i,j in it.combinations(range(4),2):
  for sign in [-1,1]:
   for w in [1,2]:
    for name,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh)]:add(f'{name}({w}*({i}+{sign}*{j}))',fn(w*(q[:,i]+sign*q[:,j])))
 return np.column_stack(fs),np.array(names)
for name,q in coords.items():
 f,names=features(q); f=(f-f[:56].mean(0))/f[:56].std(0); f=np.c_[np.ones(len(q)),f]; names=np.concatenate((['1'],names)); print('COORD',name)
 for target in [y,*z2.T]:
  used=[0,1,2,3,4]; best=[]
  for k in range(13):
   A=f[:56,used]; coef=np.linalg.lstsq(A,target[:56],rcond=None)[0]; e=target[:56]-A@coef; ev=np.mean(abs(target[56:]-f[56:,used]@coef)); best.append((ev,np.std(e),','.join(names[used[5:]]))); predres=f[:56]-A@np.linalg.lstsq(A,f[:56],rcond=None)[0]; scores=abs(e@predres)/np.maximum(np.linalg.norm(predres,axis=0),1e-8); scores[used]=0; used.append(np.argmax(scores));
  print('linear',best[0][:2], 'best',min(best),'final',best[-1]);

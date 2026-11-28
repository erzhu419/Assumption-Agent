import pickle,numpy as np
from itertools import combinations
from scipy.optimize import least_squares,root
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train']; t=s['eval_times'];Y=tr['traj'];y=tr['y']; n=len(y)
fit=lambda D,Y:np.linalg.lstsq(D,Y,rcond=None)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
def terms(z):
 w=np.roll(z,-1,axis=-1);return np.stack((np.ones_like(z),z,w,z*w,z**3),axis=-1)
u,v=(s['panel_a'][k] for k in ['u','v']);d=terms(u);keep=np.ones(len(u),bool)
for _ in range(6):
 c=np.array([fit(d[keep,j],v[keep,j]) for j in range(4)]);r=np.linalg.norm((d*c).sum(-1)-v,axis=-1);keep=r<6*np.median(r)
z=np.array([root(lambda z:(terms(z)*c).sum(-1)-x,x).x for x in tr['x']]);print('decode residual',np.max(abs((terms(z)*c).sum(-1)-tr['x'])))
p=s['panel_c'];u=p['u'];v=p['v2']; pairs=list(combinations(range(4),2));cross=lambda w:np.array([w[:,i]*w[:,j] for i,j in pairs]).T
coef=fit(aff(cross(u)),p['v1']); print('v1 cross coef',coef)
a=aff(cross(z))@coef
b0=s['panel_b']; h=aff(z)@fit(aff(b0['u']),b0['v']);print('B matrix singular values',np.linalg.svd(fit(aff(b0['u']),b0['v'])[1:],compute_uv=False))
def trybasis(label,f,start):
 def err(k):
  F=aff(f(u,k));return (F@fit(F,v)-v).ravel()
 o=least_squares(err,start,max_nfev=200);F=aff(f(u,o.x));C=fit(F,v);R=F@C-v; H=(np.linalg.svd(F,full_matrices=False)[0]**2).sum(1);q=np.sqrt(np.mean((R/(1-H[:,None]))**2)); print('BASIS',label,o.x,'RMSE',np.sqrt(np.mean(R**2)),'LOO',q);return (q,aff(f(z,o.x))@C)
bs=[]
for j,f,k in [('quad',lambda w,k:w+k[0]*w*w,[.45]),('cubic',lambda w,k:w+k[0]*w*w+k[1]*w**3,[.45,0]),('quartic',lambda w,k:w+k[0]*w*w+k[1]*w**3+k[2]*w**4,[.45,0,0]),('crossnext',lambda w,k:w+k[0]*w*w+k[1]*w*np.roll(w,-1,axis=1),[.45,0]),('crossprev',lambda w,k:w+k[0]*w*w+k[1]*w*np.roll(w,1,axis=1),[.45,0]),('rational',lambda w,k:(w+k[0]*w*w)/(1+k[1]*w*w),[.45,0]),('tanhquad',lambda w,k:np.tanh(k[1]*w)/k[1]+k[0]*w*w,[.45,.1]),('sinquad',lambda w,k:np.sin(k[1]*w)/k[1]+k[0]*w*w,[.45,.1]),('quad_each',lambda w,k:w+k*w*w,[.45]*4),('quad_full',lambda w,k:np.c_[w,w*w],[])]:
 if len(k):bs.append(trybasis(j,f,k))
b=min(bs,key=lambda x:x[0])[1]
T=lambda k:-np.expm1(-t[:,None]*k)
rates=least_squares(lambda k:(T(k)@fit(T(k),Y[:56].T)-Y[:56].T).ravel(),[.4,1.8]).x
amps=fit(T(rates),Y.T).T; target=np.c_[y,amps]; print('rates',rates,'target scale',target.std(0))
out=lambda x,z:(x[:,:,None]*z[:,None,:]).reshape(len(x),-1)
base={'z':z,'h':h,'a':a,'b':b,'dot':(a*b).sum(1,keepdims=True),'mix':a[:,0:1]*b[:,1:2],'ab':out(a,b),'aa':out(a,a),'bb':out(b,b),'hz':out(h,z),'ha':out(h,a),'hb':out(h,b)}
for keys in ['a b dot','a b dot mix','a b ab','z a b dot mix','z a b ab','z a b ab bb','z a b ab ha hb','z a b ab aa bb','z a b ab aa bb ha hb']:
 F=aff(np.column_stack([base[k] for k in keys.split()])); C=fit(F[:56],target[:56]);P=F@C; print('FIT',keys,'train',np.abs(P[:56]-target[:56]).mean(0),'CV',np.abs(P[56:]-target[56:]).mean(0),'traj',np.abs(P[56:,1:]@T(rates).T-Y[56:]).mean())
 if keys=='a b dot mix': print('basic coefficients',C)
for j in range(3):
 F=aff(np.column_stack([base[k] for k in ('z','a','b','dot','mix','ab','aa','bb','ha','hb')])); labels=['int']+[f'{k}{i}' for k in ('z','a','b','dot','mix','ab','aa','bb','ha','hb') for i in range(base[k].shape[1])]; P=F[:56].copy(); norms=np.linalg.norm(P,axis=0);sel=[0];
 for _ in range(9):
  pred=F[:,sel]@fit(P[:,sel],target[:56,j]);gain=abs(P.T@(target[:56,j]-pred[:56]))/norms;gain[sel]=-1;sel.append(np.argmax(gain));print('greedy',j,labels[sel[-1]],np.abs(pred[:56]-target[:56,j]).mean(),np.abs(pred[56:]-target[56:,j]).mean())

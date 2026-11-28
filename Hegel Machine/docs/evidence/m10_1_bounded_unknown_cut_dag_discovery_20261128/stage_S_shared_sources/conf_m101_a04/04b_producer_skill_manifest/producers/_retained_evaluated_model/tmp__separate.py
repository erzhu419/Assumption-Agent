import pickle,numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['eval_times']; z=s['train']['traj']; Y=np.c_[y,z]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for k in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def err(p,q): return np.r_[np.mean(abs(p[:,0]-q[:,0])),np.mean(abs(p[:,1:]-q[:,1:])),np.mean(abs(p[:,-1]-q[:,-1]))]
models={}
for key in ['panel_a','panel_b','panel_c']:
 p=s[key]
 for v in sorted(set(p)-{'u'}):
  a=p['u']; b=p[v]; F=poly(a,1); w=np.arange(len(a))
  if key=='panel_a':
   for _ in range(8):
    res=np.linalg.norm(F@fit(F[w],b[w])-b,axis=1); w=np.argsort(res)[:38]
  C=fit(F[w],b[w]); r=np.linalg.norm(F@C-b,axis=1); G=poly(a,2); D=fit(G[w],b[w]); print(key,v,'linear matrix',C.round(5).tolist(),'sv',np.linalg.svd(C[1:],compute_uv=False).round(5),'residual quantiles',np.quantile(r,[0,.5,.75,.9,1]).round(6),'quad rmse',np.mean((G[w]@D-b[w])**2)**.5,'quad rows',D[5:].round(4).tolist());models[key+v]=(C,D)
A=models['panel_av'][0]; B=models['panel_bv'][0]
vari={'x':x,'a+':poly(x,1)@A,'a-':(x-A[0])@np.linalg.inv(A[1:]),'b+':poly(x,1)@B,'b-':(x-B[0])@np.linalg.inv(B[1:])}
for n in ['a+','a-']:
 vari[n+'b+']=poly(vari[n],1)@B;vari[n+'b-']=(vari[n]-B[0])@np.linalg.inv(B[1:]);vari[n+'c']=np.c_[*[poly(vari[n],2)@models['panel_c'+v][1] for v in ['v1','v2']]]
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=150)
a=vari['a-']; print('t',t,'sv traj',np.linalg.svd(z,compute_uv=False)); U,S,V=np.linalg.svd(z,full_matrices=False);print('time modes',V[:4]); print('trajectory rank 2 mae',np.mean(abs((U[:,:2]*S[:2])@V[:2]-z),axis=0))
D=poly(a,3); co=fit(D,np.c_[y,U[:,:3]*S[:3]]); names=['const']+[''.join(map(str,ii)) for k in range(1,4) for ii in combinations_with_replacement(range(4),k)];print('latent coefficients:');print('\n'.join(f'{i:>6} {v}' for i,v in zip(names,co)))
p=s['panel_c']; u=p['u']; Q=poly(u,1)
for k in ['v1','v2']:
 v=p[k]
 for law,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rational',lambda g:g/(1+g*g))]:
  for mode in ['diag','rotated']:
   lin=fit(Q,v)[1:]; gu=u if mode=='diag' else u@lin/np.linalg.norm(lin,axis=0)
   def ff(th):return np.c_[Q,fun(gu*th[:4]+th[4:])]
   def rr(th):
    m=ff(th);return (m@fit(m,v)-v).ravel()
   b=least_squares(rr,np.r_[np.ones(4)*2,np.zeros(4)],max_nfev=90);print('C',k,law,mode,np.std(b.fun),b.x.round(3))
# Test whether a separate coordinatewise nonlinear feature for 0 and 3 resolves the dominant cubic terms.
for j,T in enumerate([y,z]):
 for law,fun in [('sin',np.sin),('tanh',np.tanh),('rational',lambda g:g/(1+g*g))]:
  for mode in ['add','interaction']:
   def ff(th,a):
    h=fun(a[:,[0,3]]*th[:2]+th[2:]); q=poly(a,2)
    return np.c_[q,h] if mode=='add' else np.c_[q,h,(h[:,:,None]*a[:,None,:]).reshape(len(a),8)]
   def rr(th):
    m=ff(th,a[:52]);return (m@fit(m,T[:52])-T[:52]).ravel()
   for init in [1,2,3]:
    b=least_squares(rr,[init,init,0,0],max_nfev=120,bounds=([.15,.15,-3,-3],[5,5,3,3])); m=ff(b.x,a); er=np.mean(abs(m[52:]@fit(m[:52],T[:52])-T[52:])); print('sep',j,law,mode,init,b.x.round(3),round(np.std(b.fun),6),round(er,6))

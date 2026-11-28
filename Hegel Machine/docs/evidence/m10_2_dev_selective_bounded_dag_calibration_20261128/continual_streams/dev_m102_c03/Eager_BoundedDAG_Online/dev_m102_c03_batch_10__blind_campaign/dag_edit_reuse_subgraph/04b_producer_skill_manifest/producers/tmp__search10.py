import numpy as np,pickle,itertools
from scipy.linalg import qr
from cal10 import amap,other,ls
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; z=amap(s)(x); tools={};exec(open('/holdout_task/_frozen_nodes/node_00.py').read(),tools)
for m in [24,36,48,52,56,60]:
 a={**s,'train':{k:v[:m] for k,v in tr.items()},'test':{'x':x[m:]}}; p=tools['run_node']({'__input__':a}); print('frozen',m,np.mean(abs(p['y_hat']-tr['y'][m:])),np.mean(abs(p['state_pred']-tr['traj'][m:])))
p=s['panel_a']; a=p['u']; b=p['v']; A=np.c_[np.ones(len(a)),a]; keep=np.linalg.norm(b-A@ls(A,b),axis=1)<4; a,b,A=a[keep],b[keep],A[keep]; np.set_printoptions(precision=5,suppress=True)
for j in range(4):
 u=a[:,j]; v=a[:,(j+1)%4]; F=np.c_[A,u*v]; r=b[:,j]-F@ls(F,b[:,j]); print('A',j,'coeff',ls(F,b[:,j]),'rms',np.std(r),'residual quantiles',np.quantile(abs(r),[.1,.5,.8,.99])); cand={f'{i},{k}':a[:,i]*a[:,k] for i,k in itertools.combinations_with_replacement(range(4),2)}
 for k in (.22,.5,1,1.25,2.15,2.4,3):
  for i,w in enumerate([u,v,u+v,u-v,u*v]):
   cand[f'sin {k} {i}']=np.sin(k*w);cand[f'cos {k} {i}']=np.cos(k*w);cand[f'rat {k} {i}']=w/(1+k*w*w);cand[f'prod {k} {i}']=u*v/(1+k*w*w)
 out=[]
 for name,q in cand.items():
  G=np.c_[F,q];out.append((np.std(b[:,j]-G@ls(G,b[:,j])), name))
 print(sorted(out)[:8])
# Compare archived sparse supports in raw and calibrated coordinates, then fit sparse feature grammars.
Y=np.c_[tr['y'],ls(-np.expm1(-s['eval_times'][:,None]*np.array([.4197,1.8143])),tr['traj'].T).T]; q=-np.expm1(-s['eval_times'][:,None]*np.array([.4197,1.8143])); ys=(2,5,10,14,16,23,24,26,33,35,40,50,51,52,54,61,63,64,65,67,77,78,85,87,92,93,96,101,102,104,106,111,119,128,132,137); fs=(7,9,10,24,30,31,35,36,39,40,51,54,61,68,71,72,85,86,96,101,104,106,111,112,123,127,128,134)
for lab,h in [('raw',x),('z',z)]:
 a,g=tools['_features'](h); G=np.column_stack(a+g+[np.prod(h[:,j],axis=1) for j in itertools.combinations(range(4),3)]); n=52
 for name,H in [('archive',G),('with_fourier',np.c_[G,*[f(k*h) for k in (.7,1,1.25,1.5,1.8,2.15,2.4,2.8,3.2,4,5) for f in (np.sin,np.cos,np.tanh)]])]:
  for cols in [ys,fs]:
   if name=='archive':
    F=H[:,[0]+[i+1 for i in cols]];e=F[n:]@ls(F[:n],Y[:n])-Y[n:];print(lab,'support',np.mean(abs(e),0),np.mean(abs(e[:,1:]@q.T)))
  for mode in range(3):
   F=H[:,1:];F=F-F[:n].mean(0);F/=np.maximum(np.linalg.norm(F[:n],axis=0),1e-10); t=Y[:,mode]-Y[:n,mode].mean(); ids=[]; r=t[:n]; best=[]
   for k in range(32):
    score=abs(F[:n].T@r); score[ids]=-1;ids.append(np.argmax(score)); B=F[:,ids]; c=ls(B[:n],t[:n]);r=t[:n]-B[:n]@c;best.append((np.mean(abs(t[n:]-B[n:]@c)),k+1,np.std(r)))
   print(lab,name,mode,sorted(best)[:3],ids[:14])

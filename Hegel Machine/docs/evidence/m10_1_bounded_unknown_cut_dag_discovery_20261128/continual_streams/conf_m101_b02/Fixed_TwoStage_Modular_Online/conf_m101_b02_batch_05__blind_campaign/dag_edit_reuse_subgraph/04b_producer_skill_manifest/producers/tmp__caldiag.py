import pickle, numpy as np
from itertools import combinations_with_replacement
with open('/holdout_task/public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=140)
def dictionary(u):
 d={'1':np.ones(len(u))}; d.update({str(i):v for i,v in enumerate(u.T)})
 for k in (2,3):
  for inds in combinations_with_replacement(range(4),k): d[''.join(map(str,inds))]=np.prod(u[:,inds],axis=1)
 for i,v in enumerate(u.T):
  for name,fn in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('sin2',lambda x:np.sin(2*x)),('abs',np.abs),('rat',lambda x:x/(1+x*x))]: d[name+str(i)]=fn(v)
 return d
def diag(label,u,v):
 print('\n',label,'ranges',u.min(0),u.max(0),v.min(0),v.max(0)); D=dictionary(u); a=np.stack(list(D.values())[:5],axis=1); w=np.ones(len(u))
 if label=='a':
  a=np.column_stack((a,u**3));
  for k in range(35):
   q=np.sqrt(w); b=np.linalg.lstsq(a*q[:,None],v*q[:,None],rcond=None)[0]; r=np.linalg.norm(a@b-v,axis=1); w=1/(1+(r/max(.003,1.5*np.median(r)))**2)
  idx=np.argsort(r)[-3:]; print('largest',idx,r[idx], 'median',np.median(r)); w[idx]=0
 else: idx=[]
 keep=w>0; print('affine coeff',np.linalg.lstsq(a[keep,:5],v[keep],rcond=None)[0]);
 for j in range(v.shape[1]):
  y=v[keep,j]; names=list(D)[:5]; rem=list(D)[5:]; steps=[]
  for k in range(8):
   x=np.column_stack([D[n][keep] for n in names]); b=np.linalg.lstsq(x,y,rcond=None)[0]; r=y-x@b; h=np.sum(np.linalg.qr(x,mode='reduced')[0]**2,axis=1); steps.append((names[-1],round(np.sqrt(np.mean(r*r)),6),round(np.mean(np.abs(r/np.maximum(1-h,1e-9))),6),round(b[-1],5)))
   z=np.column_stack([D[n][keep] for n in rem]); z-=x@np.linalg.lstsq(x,z,rcond=None)[0]; score=np.abs(r@z)/np.maximum(np.linalg.norm(z,axis=0),1e-9); names.append(rem.pop(np.argmax(score)))
  print(j,steps)
for k,key,u in [('a','v','u'),('b','v','u'),('c','v1','u'),('c','v2','u'),('c','v2','v1')]:
 p=s['panel_'+k]; diag(k+(' '+u+'->'+key if k=='c' else ''),p[u],p[key])
t=s['train']; print('traj svd',np.linalg.svd(t['traj'],compute_uv=False)); print('x range',t['x'].min(0),t['x'].max(0)); print('telemetry',s['telemetry_vec']); print('traj time',s['eval_times'])

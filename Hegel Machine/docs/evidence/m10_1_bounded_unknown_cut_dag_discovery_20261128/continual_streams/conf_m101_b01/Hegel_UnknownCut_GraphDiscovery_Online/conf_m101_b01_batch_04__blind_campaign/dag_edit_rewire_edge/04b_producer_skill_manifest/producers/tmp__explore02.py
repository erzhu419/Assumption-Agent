import pickle, numpy as np
from itertools import combinations_with_replacement, combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=160)
def load(i):
 ns={};exec(open(f'/holdout_task/_frozen_nodes/node_0{i}.py').read(),ns);return ns['run_node']
a=load(0)({'__input__':s}); u=a['calibrated_x']; z=load(1)({'node_00':a})['x']; n=len(s['train']['y']); Y=np.column_stack([s['train']['y'],s['train']['traj']]); t=s['eval_times']
def poly(x,p):
 d=x.shape[1];return np.column_stack([np.prod(x[:,k],axis=1) for j in range(p+1) for k in combinations_with_replacement(range(d),j)])
def loo(A,Y):
 c,_,rank,_=np.linalg.lstsq(A,Y,rcond=1e-11); h=np.sum(np.linalg.svd(A,full_matrices=False)[0][:,:rank]**2,axis=1); e=A@c-Y; return np.mean(np.abs(e),axis=0),np.mean(np.abs(e/np.maximum(1e-9,1-h[:,None])),axis=0)
print('cal to stage1',np.linalg.lstsq(poly(u,1),z,rcond=None)[0]); print('traj svd',np.linalg.svd(Y[:,1:],compute_uv=False));print('cal range',u.min(axis=0),u.max(axis=0),'trans range',z.min(axis=0),z.max(axis=0))
p=s['panel_c']; q=p['u'];
for k in ['v1','v2']:
 for j in range(4):
  D=poly(q,2); b=p[k][:,j]; out=[]
  for k2 in range(4):
   best=(1e9,None,None)
   for inds in combinations(range(5,15),k2):
    A=D[:,list(range(5))+list(inds)]; r,l=loo(A,b[:,None]);
    if l[0]<best[0]:best=(l[0],inds,r[0]);
   out.append(best)
  print(k,j,'sparse polynomial',out)
for name,x in [('calibrated',u),('transformed',z)]:
 for degree in range(1,5):
  A=poly(x[:n],degree); r,l=loo(A,Y); print(name,degree,'train y/time',r[0],r[1:].mean(),'LOO',l)
for freq in [0.5,1,1.5,2]:
 A=np.column_stack([poly(z,2),np.sin(freq*u),np.cos(freq*u),np.sin(freq*z),np.cos(freq*z)])[:n]; print('trig',freq,loo(A,Y)[1])

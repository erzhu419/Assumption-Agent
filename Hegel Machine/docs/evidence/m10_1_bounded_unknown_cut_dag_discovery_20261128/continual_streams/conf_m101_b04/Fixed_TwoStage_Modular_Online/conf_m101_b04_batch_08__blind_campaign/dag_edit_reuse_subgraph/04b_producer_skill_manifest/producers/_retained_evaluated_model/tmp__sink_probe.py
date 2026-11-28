import sys,pickle,itertools as it,numpy as np
from scipy.linalg import qr
sys.path.insert(0,'/holdout_task/_frozen_nodes'); from node_00 import run_node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=run_node({'__input__':s}); Y=np.c_[d['y'],d['traj']]; N=len(Y)
def poly(x,p):
 return np.column_stack([np.prod(x[:,j],axis=1) for k in range(p+1) for j in it.combinations_with_replacement(range(x.shape[1]),k)])
def grammar(x):
 fs,names=[],[]
 def add(z,name): fs.append(z);names.append(name)
 for j in it.combinations_with_replacement(range(4),3): add(np.prod(x[:,j],axis=1),'cube'+str(j))
 dirs=[(x[:,j],str(j)) for j in range(4)]+[(x[:,i]+a*x[:,j],f'{i}+{a}*{j}') for i,j in it.combinations(range(4),2) for a in (-2,-1,-.5,.5,1,2)]
 for z,name in dirs:
  for a in (.5,1,1.5,2,3):
   for fun in (np.sin,np.cos,np.tanh): add(fun(a*z),f'{fun.__name__}({a}*({name}))')
  for a in (.25,.5,1,2):
   q=1/(1+a*z*z); add(q,f'Q({a},{name})'); add(z*q,f'({name})*Q({a},{name})')
   if len(name)==1:
    for j in range(4):
     if str(j)!=name: add(x[:,j]*q,f'{j}*Q({a},{name})')
 return np.array(fs).T,names
def paths(B,F,y,a,v,steps=8):
 q=qr(B[a],mode='economic')[0]; R=F[a]-q@(q.T@F[a]); norm=np.maximum(np.linalg.norm(R,axis=0),1e-12); inds=[]; out=[]; fit=B
 for i in range(steps+1):
  C=np.linalg.lstsq(fit[a],y[a],rcond=1e-9)[0]; r=y[a]-fit[a]@C
  out.append((np.mean(abs(y[v]-fit[v]@C)),np.sqrt(np.mean(r*r)),tuple(inds)))
  score=np.abs(r@R/norm); score[inds]=-1; inds.append(np.argmax(score)); fit=np.c_[B,F[:,inds]]
 return out
if __name__=='__main__':
 print('summary',s['train']['summary'],'telemetry',s['telemetry_vec'],'traj spectrum',np.linalg.svd(d['traj']-d['traj'].mean(0),compute_uv=False))
 folds=list(np.array_split(np.random.default_rng(30).permutation(N),6))+[np.arange(54,N)]
 for k in ('xa','xab','z','xs','raw'):
  x=d[k] if k in d else s['train']['x']; F,names=grammar(x)
  for deg in (1,2):
   B=poly(x,deg)
   for y,lab in ((Y[:,0],'y'),(Y[:,1:]@np.linalg.svd(Y[:,1:],full_matrices=False)[2][0],'trajPC')):
    runs=[paths(B,F,y,np.setdiff1d(np.arange(N),v),v) for v in folds]; m=np.array([[t[0] for t in r] for r in runs]); j=np.argmin(m[:6].mean(0)); print(k,deg,lab,'k',j,'mean',round(m[:6,j].mean(),4),'max',round(m[:6,j].max(),4),'tail',np.round(m[-1],4),'full_terms',[names[i] for i in paths(B,F,y,np.arange(N),np.arange(N))[j][2]])
 p=s['panel_c']; x=p['u']; F,names=grammar(x); B=poly(x,1)
 for lab in ('v1','v2'):
  for y in p[lab].T:
   r=paths(B,F,y,np.arange(len(y)),np.arange(len(y)),5); print('panel',lab,'affine/quad rms',round(r[0][1],4), round(np.std(y-poly(x,2)@np.linalg.lstsq(poly(x,2),y,rcond=None)[0]),4),'sparse rms',np.round([v[1] for v in r],4),[names[i] for i in r[-1][2]])

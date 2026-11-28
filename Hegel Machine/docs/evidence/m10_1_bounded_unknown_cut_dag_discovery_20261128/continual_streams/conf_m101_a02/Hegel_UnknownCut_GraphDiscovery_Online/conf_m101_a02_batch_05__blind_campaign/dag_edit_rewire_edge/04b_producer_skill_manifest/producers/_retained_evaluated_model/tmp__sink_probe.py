import runpy,pickle,numpy as np,itertools as it
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a,b=[runpy.run_path('/holdout_task/_frozen_nodes/node_%02d.py'%i)['run_node'] for i in range(2)]
c=a({'__input__':s}); z=b({'node_00':c})['z_tr']; x=c['cal_tr']; t=s['eval_times']; Y=np.column_stack([s['train']['y'],s['train']['traj']]); print('times',t,'x/z ranges',np.ptp(x,axis=0),np.ptp(z,axis=0)); print('trajectory sv',np.linalg.svd(Y[:,1:],compute_uv=False)); print('traj / time median',np.median(Y[:,1:]/t,axis=0));
N=len(x); folds=[(np.arange(52),np.arange(52,N))]; folds += [(np.setdiff1d(np.arange(N),v),v) for v in np.array_split(np.random.default_rng(7).permutation(N),4)]
def poly(A,d):
 return np.column_stack([np.ones(N)]+[np.prod(A[:,j],axis=1) for k in range(1,d+1) for j in it.combinations_with_replacement(range(A.shape[1]),k)])
def assess(label,B):
 scores=[]
 for i,j in folds:
  pp=B[j]@np.linalg.lstsq(B[i],Y[i],rcond=1e-9)[0]; e=np.mean(abs(pp-Y[j]),axis=0); scores.append([e[0],np.mean(e[1:]),np.max(e[1:])]);
 fit=np.mean(abs(Y-B@np.linalg.lstsq(B,Y,rcond=1e-9)[0]),axis=0); print(label,B.shape[1],np.round(scores[0],5),np.round(np.max(scores[1:],axis=0),5),'train',np.round(fit[[0,1,-1]],5))
for label,A in [('x',x),('z',z),('xz',np.c_[x,z])]:
 for d in range(1,3 if label=='xz' else 4): assess(label+str(d),poly(A,d))
J=np.einsum('ni,nj->nij',x,z).reshape(N,-1)
assess('x z x*z',np.c_[np.ones(N),x,z,J]); assess('z + x*z',np.c_[np.ones(N),z,J]); assess('z2 + x*z',np.c_[poly(z,2),J]); assess('z2 + matched cross',np.c_[poly(z,2),x*z]); assess('linear + matched cross',np.c_[np.ones(N),x,z,x*z]);
print('traj examples',np.round(Y[:6,1:],4))

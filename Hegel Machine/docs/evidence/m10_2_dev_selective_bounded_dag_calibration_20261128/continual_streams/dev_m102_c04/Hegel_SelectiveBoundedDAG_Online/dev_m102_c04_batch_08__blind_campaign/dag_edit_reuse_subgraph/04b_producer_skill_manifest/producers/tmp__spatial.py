import pickle,runpy,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement,product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c04_batch_00_solver.py'); A=a['_affine']; L=a['_ls']; C=s['panel_c']; u=C['u']; pairs=list(combinations_with_replacement(range(4),2));
def Q(x):return A(np.c_[x,*[x[:,i]*x[:,j] for i,j in pairs]])
def R(x,k):return A(np.c_[x,np.cos(k*x)])
w=L(Q(u),C['v1']); k=minimize_scalar(lambda k: np.linalg.norm(C['v2']-R(u,k)@L(R(u,k),C['v2'])),bounds=(1.5,3.2),method='bounded').x; v=L(R(u,k),C['v2']); print('quadratic',np.round(w,4),'cos k',k,'cos coefficients',np.round(v,4));
x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; fz=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py'); methods={};
for key,lab in [('panel_a','A'),('panel_b','B')]:
 b,m=a['_calibration'](s[key]); methods[lab+'+']=lambda z,b=b,m=m:z@m+b; methods[lab+'-']=lambda z,b=b,m=m:(z-b)@np.linalg.inv(m)
trans={'I':x};
for n in (1,2,3):
 for seq in product(methods,repeat=n):
  if any(j[0]==l[0] and j!=l for j,l in zip(seq,seq[1:])):continue
  z=x
  for j in seq:z=methods[j](z)
  trans[' '.join(seq)]=z
ranks=[[],[]]; folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),ii),ii) for ii in np.array_split(np.random.default_rng(42).permutation(72),4)]; Bs=[fz['_temporal_basis'](s['eval_times'],t[i]) for i,j in folds];
for name,z in trans.items():
 p,r=Q(z)@w,R(z,k)@v; cross=(p[:,:,None]*r[:,None,:]).reshape(-1,16); common=np.c_[p,r,np.sum(p*r,axis=1)]; F={'common':common,'full':np.c_[p,r,cross],'diag':np.c_[p,r,p*r],'channels':np.c_[p,r],'raw':np.c_[z,common],'expanded':np.c_[z,p,r,cross],'balanced':np.c_[p,r,np.sum(p*r,axis=1),p+r,p-r]}
 for key,ff in F.items():
  D=A(ff); sy=[]; st=[]
  for (i,j),B in zip(folds,Bs):
   sy.append(abs(D[j]@L(D[i],y[i])-y[j]).mean()); st.append(abs(D[j]@L(D[i],t[i]@B)@B.T-t[j]).mean())
  for vals,rank in zip([sy,st],ranks):rank.append((max(vals),vals[0],name,key,round(np.mean(vals),5)))
for rank in ranks:print('best worst/first/mean',*sorted(rank)[:16],sep='\n');print('best first',*sorted(rank,key=lambda x:x[1])[:5],sep='\n')

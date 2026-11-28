exec(open('/tmp/readout_diag.py').read().split('for keys in')[0])
import itertools
from scipy.optimize import least_squares
from scipy.linalg import svd
u=s['panel_b']['u'];v=s['panel_b']['v'];q=u*u; p=m._cycle(u)
for name,extra in [('none',q[:,:0]),('sq',q),('cycle',p)]:
 def bfit(f,ix):
  d=np.c_[m._affine(u),extra,np.sin(u*f)]; return d@m._fit(d[ix],v[ix])-v
 for independent in [False,True]:
  ds=[]; full=np.arange(len(u));f0=np.full(4 if independent else 1,2.84)
  for i in range(-1,len(u)):
   ix=full if i<0 else full[full!=i]; f=least_squares(lambda f:bfit(f,ix)[ix].ravel(),f0).x; r=bfit(f,ix)
   if i<0: print('B',name,independent,f,'fit',np.std(r),flush=True)
   else:ds.extend(r[i]);
  print('  nested loo',np.std(ds),flush=True)
base={'z':z,'b':b,'c':c1,'d':c2}; trU,trS,trV=svd(tr,full_matrices=False); target=np.c_[y,trU[:,:2]*trS[:2]]; print('traj basis',trV[:2]);print('target corr',np.corrcoef(target.T)); print('traj avg',tr.mean(0));
F={}
for key,A in base.items():
 for i,a in enumerate(A.T): F[key+str(i)]=a
 for i,a in enumerate(A.T):
  for j,bb in enumerate(A.T):
   if i<=j:F[key+str(i)+key+str(j)]=a*bb
for (k,a),(l,bb) in itertools.combinations(base.items(),2):
 if k=='z' and l=='c' or k in 'bc' and l in 'cd' or k=='z' and l in 'bd':
  for i,j in itertools.product(range(4),repeat=2):F[k+str(i)+l+str(j)]=a[:,i]*bb[:,j]
for k,A in base.items():
 for i,a in enumerate(A.T):
  for f in [.5,1,1.5,2,2.5,3,4]:
   for n,fn in [('s',np.sin),('o',np.cos),('t',np.tanh)]: F[f'{n}{f}{k}{i}']=fn(f*a)
H=np.column_stack(list(F.values()));H-=H.mean(0);H/=np.maximum(1e-10,np.std(H,axis=0)); names=list(F)
for j in range(3):
 Y=target[:,j];r=Y-Y.mean(); ids=[];print('target',j,np.std(Y));
 for step in range(16):
  D=np.c_[np.ones(len(Y)),H[:,ids]]; Q=np.linalg.qr(D)[0];R=H-Q@(Q.T@H);k=np.argmax(np.abs(R.T@r)/np.maximum(1e-8,np.linalg.norm(R,axis=0)));ids.append(k);D=np.c_[D,H[:,k]];res=Y-D@m._fit(D,Y);r=res; print(names[k],round(np.std(r),5),end='; ')
 print()
for keys in ['Z','ZBCD','ZBCD+square','ZBCD+square+interaction','Z+polynomial3','Z+polynomial4']:
 D0=np.c_[z,b,c1,c2]; d={'Z':z,'ZBCD':D0,'ZBCD+square':np.c_[D0,D0**2],'ZBCD+square+interaction':np.c_[D0,H[:,:136]]}.get(keys)
 if d is None:
  deg=int(keys[-1]);d=np.column_stack([np.prod(z[:,inds],axis=1) for k in range(1,deg+1) for inds in itertools.combinations_with_replacement(range(4),k)])
 D=m._affine(d); scores=[]
 for it,iv in folds:
  ix=np.r_[it,iv];a=sink.run_node({'node_00':{'g':D[ix],'fd':b[ix],'cd':np.c_[c1,c2][ix],'y':y[it],'traj':tr[it]}});scores.append([np.mean(abs(a['y_hat']-y[iv])),np.mean(abs(a['state_pred']-tr[iv]))]);
 print(keys,'official',np.round(scores[0],4),'worst',np.round(np.max(scores,axis=0),4))

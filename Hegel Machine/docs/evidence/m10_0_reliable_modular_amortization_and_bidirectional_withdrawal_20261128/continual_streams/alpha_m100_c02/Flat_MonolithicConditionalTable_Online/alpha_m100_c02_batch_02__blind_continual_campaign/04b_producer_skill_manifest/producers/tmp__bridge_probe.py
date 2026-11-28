import pickle,numpy as np,itertools
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; g=np.linalg.norm(y,axis=1)<12; sw=(x!=0).sum(1)==1; n=len(x)
np.set_printoptions(precision=4,suppress=True)
print('axis true by direction; response rows'); print(np.c_[x[sw],y[sw]])
s=c['single_channel_sweeps']; offs=[]; diag=[]; cubic=[]
for j in range(4):
 t=s['lab_id']==j; u=s['true_stress'][t]; v=s['cmd_stress'][t]; k=abs(v)<6; b=np.linalg.lstsq(np.c_[np.ones(sum(k)),u[k],u[k]**3],v[k],rcond=None)[0]; offs.append(b[0]);diag.append(b[1]); cubic.append(b[2]);
a=x+.18*x**3; o=np.array(offs); d=np.array(diag); q=np.array(cubic); p0=np.linalg.lstsq(a[g],y[g]-o,rcond=None)[0]; p0[np.diag_indices(4)]=d
print('fixed sweep coefficients',o,d,q,'initial M',p0)
models={}
for deg in (1,2,3):
 terms=[ij for k in range(1,deg+1) for ij in itertools.combinations_with_replacement(range(4),k)]; models['polynomial '+str(deg)]=np.c_[np.ones(n),np.column_stack([np.prod(x[:,ij],axis=1) for ij in terms])]
models['premixed cubic']=np.c_[np.ones(n),a]; models['nonlinear diag only']=np.c_[np.ones(n),x]
for name,D in models.items():
 yy=y-x**3*q if name=='nonlinear diag only' else y
 e=[]; errs=[]
 for i in np.flatnonzero(g):
  m=g&(np.arange(n)!=i); b=np.linalg.lstsq(D[m],yy[m],rcond=None)[0]; e.append(np.linalg.norm(D[i]@b-yy[i])); errs.append(D[i]@b-yy[i]);
 print(name,'LOO',np.quantile(e,[.5,.9,1]));
 if name=='polynomial 3':
  b=np.linalg.lstsq(D[g],yy[g],rcond=None)[0]; print('train',np.max(np.linalg.norm(D[g]@b-yy[g],axis=1))); print('terms');
  for ij,z in zip([()]+terms,b): print(ij,z)
cs=np.eye(4,dtype=bool); cross=~cs
for kind in ('pre','post','diag','rawmixpre','sumdiffcub'):
 def pred(p):
  M=p0.copy(); M[cross]=p[:12]; w=x@M
  if kind=='pre': return o+a@M
  if kind=='post': return o+w+w**3*(q/d**3)
  if kind=='diag': return o+w+x**3*q
  if kind=='rawmixpre': return o+(x+x**3*(q/d))@M
  if kind=='sumdiffcub':
   r=x[:,:,None]+p[12]*x[:,None,:]; return o+w+x**3*q+np.sum((r**3-x[:,:,None]**3-(p[12]*x[:,None,:])**3)*(M*cross),axis=1)*p[13]
 z=np.r_[p0[cross],1.,.1] if kind=='sumdiffcub' else p0[cross]; b=least_squares(lambda p:(pred(p)[g]-y[g]).ravel(),z).x; e=np.linalg.norm(pred(b)[g]-y[g],axis=1); print(kind,np.quantile(e,[.5,.9,1]),b)

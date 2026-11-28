import numpy as np,pickle,sys,glob,importlib.util
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; np.set_printoptions(precision=5,suppress=True)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
u,v=[s['panel_a'][k] for k in ('u','v')]; m=np.ones(len(u),bool)
for j in range(4):
 b=ls(aff(u[m]),v[m]); r=np.linalg.norm(v-aff(u)@b,axis=1); m=r<max(.02,2*np.median(r))
print('A',b,'sorted residuals',np.sort(r)); z=(X-b[0])@np.linalg.inv(b[1:]); zdiag=(X-b[0])/b[1:].diagonal();
ops=[]
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[p]['u'],s[p][k]; a=ls(aff(u),v); print(p,k,'matrix',a,'rmse',np.std(v-aff(u)@a));ops.append(a)
cs={'x':X,'z':z,'zd':zdiag}
for j,a in enumerate(ops):cs[str(j)]=aff(z)@a
for k,c in cs.items():
 for mode in range(4):
  i,j=np.triu_indices(4,0); h=[aff(c),np.c_[aff(c),c[:,i]*c[:,j]],np.c_[aff(c),np.tanh(c)],np.c_[aff(c),np.tanh(c),c[:,i]*c[:,j]]][mode]; pred=h[52:]@ls(h[:52],Y[:52]); e=np.mean(abs(pred-Y[52:]),axis=0); print(k,mode,round(e[0],5),round(e[1:].mean(),5))
print('traj sv',np.linalg.svd(Y[:,1:],compute_uv=False));print('y correlations',np.corrcoef(Y.T)[0])
for f in glob.glob('/tmp/hegel_continual_archive/*solver.py'):
 spec=importlib.util.spec_from_file_location('m',f); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); st=dict(s,train={k:v[:52] if k!='summary' else v for k,v in s['train'].items()},test={'x':X[52:]}); o=mod.solve_batch(st); print(f,np.mean(abs(o['y_hat']-Y[52:,0])),np.mean(abs(o['state_pred']-Y[52:,1:])))

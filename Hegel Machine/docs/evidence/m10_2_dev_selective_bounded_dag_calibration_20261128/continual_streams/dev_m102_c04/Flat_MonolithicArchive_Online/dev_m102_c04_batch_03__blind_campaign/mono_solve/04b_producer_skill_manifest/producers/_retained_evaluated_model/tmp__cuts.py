import sys,pickle,numpy as np,itertools
sys.path.insert(0,'/holdout_task')
from batch_solver import _calibration,_panel_operators,_affine as aff,_ls as ls,_temporal_basis,solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];y=s['train']['y'];T=s['train']['traj'];t=s['eval_times'];B=s['panel_b'];w=ls(aff(B['u']),B['v']); C=_panel_operators(s['panel_c']);q=_calibration(s['panel_a'])(x); b=aff(q)@w; Q=_temporal_basis(t,T[:52]);Y=np.c_[y,T@Q];I=np.arange(52);J=np.arange(52,72)
pred=solve_batch(dict(s,train={k:v[:52] if k in ('x','y','traj') else v for k,v in s['train'].items()},test={'x':x[J]}));print('recalibrated archive',np.mean(abs(pred['y_hat']-y[J])),np.mean(abs(pred['state_pred']-T[J])));print('q bounds',q.min(0),q.max(0))
D=dict(x=x,q=q,b=b,b0=aff(x)@w,pc=C[0](q),rc=C[1](q),pb=C[0](b),rb=C[1](b),p0=C[0](x),r0=C[1](x)); D.update({k+'B':aff(D[k])@w for k in ('pc','rc')});
res=[]
def add(n,f):
 A=aff(f); pred=A[J]@ls(A[I],Y[I]); e=pred-Y[J]; tr=A[I]@ls(A[I],Y[I])-Y[I];res.append((np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]@Q.T)),n,np.std(tr,axis=0)))
for k,f in D.items():
 add(k,f); add(k+' square',np.c_[f,f*f]);add(k+' quadratic',np.c_[f,*[f[:,j]*f[:,i] for j in range(4) for i in range(j,4)]])
for (k,f),(l,g) in itertools.combinations(D.items(),2):
 dot=np.sum(f*g,axis=1);add(k+' '+l,np.c_[f,g]);add(k+' '+l+' dot',np.c_[f,g,dot]);add(k+' '+l+' diag',np.c_[f,g,f*g]);add(k+' '+l+' cross',np.c_[f,g,(f[:,:,None]*g[:,None,:]).reshape(-1,16)])
print('Y BEST');print(*sorted(res,key=lambda r:r[0])[:16],sep='\n');print('TRAJ BEST');print(*sorted(res,key=lambda r:r[1])[:16],sep='\n')

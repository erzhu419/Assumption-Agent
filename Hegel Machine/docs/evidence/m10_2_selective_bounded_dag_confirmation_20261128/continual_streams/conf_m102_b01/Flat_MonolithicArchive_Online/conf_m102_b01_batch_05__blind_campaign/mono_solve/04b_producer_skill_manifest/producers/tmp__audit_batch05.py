import importlib.util
import pickle
from itertools import combinations
import numpy as np
from scipy.optimize import least_squares
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study'];t=s['train'];x,y,Y=t['x'],t['y'],t['traj']; ids=np.arange(len(y)); n=52
a,b,c=[s['panel_'+k] for k in 'abc']; A=m._aff; ls=m._ls
z=m._inverse(m._calibration(a),x); w=A(m._dead(z))@ls(A(m._dead(b['u'])),b['v']); q=m._latent(s,x); wlin=A(z)@ls(A(b['u']),b['v'])
print('Public field names:',sorted(s), 'training fields:',sorted(t))
for name,panel,out,d0,d1 in [('B',b,b['v'],A(b['u']),A(m._dead(b['u']))),('C1',c,c['v1'],A(c['u']),m._cross(c['u']))]:
    def loo(d):
        return np.sqrt(np.mean([(d[k]@ls(d[np.arange(len(d))!=k],out[np.arange(len(d))!=k])-out[k])**2 for k in range(len(d))]))
    print(name,'panel LOO RMSE affine / structured',loo(d0),loo(d1))
def softplus(z,k):return A(np.logaddexp(0,k*z))
k=least_squares(lambda b:(softplus(c['u'],b[0])@ls(softplus(c['u'],b[0]),c['v2'])-c['v2']).ravel(),[2.]).x[0]
sc=ls(softplus(c['u'],k),c['v2']); cb=ls(m._cross(c['u']),c['v1']); d=A(q)
pair=lambda v: np.c_[A(v),v[:,0]*v[:,2],v[:,1]*v[:,3]]
quad=np.c_[d,*[q[:,j]*q[:,k] for j in range(4) for k in range(j,4)]]
base=pair(q)
blocks=[('prescribed',ids[:n],ids[n:])]; rng=np.random.default_rng(12209)
for j in range(12):
    ix=rng.permutation(ids);blocks.append(('random',ix[:n],ix[n:]))
for col in [*x.T,*q.T,np.linalg.norm(q-q.mean(0),axis=1)]:
    ix=np.argsort(col);blocks.extend([('directional',ix[:n],ix[n:]),('directional',ix[20:],ix[:20])])
def err(F,target,i,j): return np.mean(abs(F[j]@ls(F[i],target[i])-target[j]))
mods={'current':base,'linear readout':d,'full quadratic':quad,'affine B':pair(m._cross(wlin)@cb),'affine C1':pair(A(w)@ls(A(c['u']),c['v1'])),'insert C2':pair(softplus(q,k)@sc)}
# Screen quotient/pairwise grammars using only the training side of each outer split.
cands=[base]; names=['none']
for j,l in combinations(range(4),2):
    for v in (0.5,1.,2.,-.5,-1.,-2.):
        delta=q[:,j]+v*q[:,l]; ds=delta**2
        cands.append(np.c_[base,ds]);names.append('pair square')
        for alpha in (0.25,1.,4.):
            r=1/(1+alpha*ds)
            cands.extend((base*r[:,None],np.c_[base,r],np.c_[base,delta*r]));names.extend(('global rational','rational residual','signed rational'))
for j,l,h in combinations(range(4),3):
    cands.append(np.c_[base,q[:,j]*q[:,l]*q[:,h]]);names.append('triple')
from collections import Counter
sums=[]; chosen=Counter(); state=[]
for _,i,j in blocks:
    fs=np.array_split(rng.permutation(i),4)
    score=[np.mean([err(F,y,np.setdiff1d(i,v),v) for v in fs]) for F in cands]; best=np.argmin(score);chosen[names[best]]+=1
    sums.append([*[err(F,y,i,j) for F in mods.values()],err(cands[best],y,i,j)])
    ph=m.solve_batch(dict(s,train=dict(x=x[i],y=y[i],traj=Y[i]),test=dict(x=x[j])))['state_pred']; state.append(np.mean(abs(ph-Y[j]),axis=0))
print('Scalar: prescribed / mean outer / worst outer MAE')
for k,v in zip([*mods,'nested grammar challenger'],np.asarray(sums).T):print(k,np.round([v[0],v.mean(),v.max()],7))
print('Nested grammar selections',chosen)
print('Worst trajectory MAE per horizon',np.round(np.max(state,axis=0),7))
print('Trajectory max mean horizon-aggregate MAE',np.mean(state,axis=1).max())

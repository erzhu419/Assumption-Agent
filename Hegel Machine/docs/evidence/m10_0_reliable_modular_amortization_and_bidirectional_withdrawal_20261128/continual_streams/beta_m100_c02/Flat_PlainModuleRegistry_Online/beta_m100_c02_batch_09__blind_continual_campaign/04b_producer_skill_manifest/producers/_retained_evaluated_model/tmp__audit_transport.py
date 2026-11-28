import sys, pickle, itertools
sys.path.insert(0, '/holdout_task')
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import _verified_upstream as up
import downstream_solver as dn
s=up.calibrate_upstream(pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'])
nx,ny=s['nx'],s['ny']; y,z=s['y_obs_raw'],s['Z_nuisance']; n=len(s['source_modes']); ns=len(s['sensor_ij']); shift,site=np.divmod(np.arange(len(y)),ns)
i,j=s['sensor_ij'].T; sensor=i+nx*j
rhs=np.column_stack([m.ravel(order='F') for m in s['source_modes']]); ix,iy=sparse.eye(nx,format='csc'),sparse.eye(ny,format='csc'); eye=sparse.eye(nx*ny,format='csc'); reaction=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
# Separate wall exchange, endpoint neighbor flux, and seam flux instead of varying them together.
params=[(0,0,0),(0,1,0),(0,1,.5),(0,1,1),(.1,0,0),(.25,0,0),(.5,0,0),(1,0,0),(0,.5,.5),(0,0,.5),(0,0,1)]
def axis(d, p):
    exchange,neighbor,seam=p; k,v=s['kappa_'+d],s['vel_'+d]; h=s['L'+d]/(len(k)-1)
    c=dn.axis_operator(k,v,h,False); w=dn.axis_operator(k,v,h,True); delta=(w-c).toarray(); out=c.toarray()
    for a,b,t in [(0,1,-1),(-1,-2,0)]:
        out[a,b]+=neighbor*delta[a,b]; out[a,t]+=seam*delta[a,t]; out[a,a]-=neighbor*delta[a,b]+seam*delta[a,t]
        out[a,a]+=exchange*s['robin_'+d]/h
    return sparse.csc_matrix(out)
ax=[sparse.kron(iy,axis('x',p),format='csc') for p in params]; ay=[sparse.kron(axis('y',p),ix,format='csc') for p in params]
models=list(itertools.product(range(len(params)),repeat=2)); responses=[]
for a,b in models:
    L=ax[a]+ay[b]+reaction
    responses.append(np.concatenate([splu(L+mu*eye).solve(rhs)[sensor] for mu in s['mu_shifts']]))
R=np.asarray(responses); fixed=models.index((3,0)); closed=models.index((0,0)); base=models.index((7,7)); np.testing.assert_allclose(R[fixed],dn.chamber_response(s)[0],atol=1e-12)
def evaluate(train,test,scale):
    Z=z[train]; P=np.linalg.solve(Z.T@Z+scale*s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T); Y=y[train]; b=Y-Z@(P@Y); B=R[:,train]; A=B-Z@(P@B); H=A.transpose(0,2,1)@A+scale*s['source_ridge']*np.eye(n); f=-(A.transpose(0,2,1)@b)
    w=np.array([dn.box_simplex(h,l,s['u_box'],s['total_budget'])[0] for h,l in zip(H,f)]); err=Y-(B@w[:,:,None]).squeeze(-1); gamma=err@P.T
    return ((R[:,test]@w[:,:,None]).squeeze(-1)+gamma@z[test].T-y[test])**2
for scale in (1.,.01):
    for name,group in [('shift',shift),('site',site),('x-wall region',np.where(sensor[site]%nx==0,0,np.where(sensor[site]%nx==nx-1,1,2)))]:
        folds=np.unique(group); scores=[]; nested=[]; picks=[]
        for g in folds:
            test=group==g; train=~test; obs=evaluate(train,test,scale); scores.append(obs)
            # Inner site groups keep measurements of the same site together.
            # Pair sites deterministically into four folds to keep the audit small.
            sub=site%4; inner=[]
            for h in np.unique(sub[train]):
                inner.append(evaluate(train&(sub!=h),train&(sub==h),scale))
            pick=int(np.argmin(np.concatenate(inner,axis=1).mean(1))); picks.append(models[pick]); nested.append(obs[pick])
        pooled=np.sqrt(np.concatenate(scores,axis=1).mean(1)); worst=np.max([np.sqrt(a.mean(1)) for a in scores],axis=0)
        print(name,scale,'fixed/closed/baseline',[(models[i],round(pooled[i],6),round(worst[i],6)) for i in (fixed,closed,base)],'top',[(models[i],round(pooled[i],6),round(worst[i],6)) for i in np.argsort(pooled)[:4]],'nested',round(np.sqrt(np.concatenate(nested).mean()),6),round(max(np.sqrt(a.mean()) for a in nested),6),'picks',sorted(set(picks)),flush=True)

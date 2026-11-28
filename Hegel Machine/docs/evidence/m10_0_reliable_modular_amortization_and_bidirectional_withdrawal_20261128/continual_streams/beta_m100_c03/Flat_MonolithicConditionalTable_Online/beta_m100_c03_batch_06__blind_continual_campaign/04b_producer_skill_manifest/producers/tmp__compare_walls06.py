import importlib.util, pickle, numpy as np

def load(name):
    p=f'/tmp/hegel_continual_archive/beta_m100_c03_batch_{name}_solver.py'
    spec=importlib.util.spec_from_file_location('a'+name,p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
m=load('05'); periodic=load('02')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
y=m._calibrate(s,s['y_obs_raw']); z=s['Z_nuisance']; sm=m._profile(s,y)
print('calibrated range',y.min(),y.max())
for name in ('robin','sealed','periodic','sealed-x','sealed-y','periodic-x','periodic-y'):
    def axis(k,v,h,r):
        isx=len(k)==s['nx']; tag='x' if isx else 'y'
        if name in ('periodic','periodic-'+tag): return periodic._axis(k,v,h)
        if name in ('sealed','sealed-'+tag): r=0.
        return orig(k,v,h,r)
    orig=m._axis; m._axis=axis
    S,U=m._responses(s); m._axis=orig
    D=m._profile(s,S); w,nu=m._box_qp(D.T@D+s['source_ridge']*np.eye(D.shape[1]), -(D.T@sm), s['u_box'],s['total_budget'])
    ij=s['sensor_ij']; wall=np.any((ij==0)|(ij==[s['nx']-1,s['ny']-1]),axis=1)
    errors=[]; ridge=s['source_ridge']; n=len(ij)
    for j in range(n//4):
        take=np.tile(np.arange(n)//4==j,len(s['mu_shifts'])); train=~take
        a=np.column_stack((S,z)); A=a[train]; rr=np.r_[np.full(S.shape[1],ridge),np.full(z.shape[1],s['nuisance_ridge'])]; k=S.shape[1]
        M=A.T@A+np.diag(rr); b=A.T@y[train]; sub=np.linalg.solve(M[k:,k:],np.column_stack((M[k:,:k],b[k:]))); W,_=m._box_qp(M[:k,:k]-M[:k,k:]@sub[:,:k],-(b[:k]-M[:k,k:]@sub[:,k]),s['u_box'],s['total_budget']); g=sub[:,k]-sub[:,:k]@W; residual=y[take]-a[take]@np.r_[W,g]; errors.append((np.sqrt(np.mean(residual**2)),max(abs(residual))))
    print(name,'J',.5*(np.linalg.norm(D@w-sm)**2+ridge*(w@w)),'rmse shift',np.sqrt(np.mean((D@w-sm).reshape(-1,n)**2,axis=1)).round(6),'wall/interior',np.sqrt(np.mean((D@w-sm).reshape(-1,n)[:,wall]**2)),np.sqrt(np.mean((D@w-sm).reshape(-1,n)[:,~wall]**2)), 'CV worst',np.max(errors,axis=0).round(6),'CV mean',np.mean(errors,axis=0).round(6),'w',w.round(5))

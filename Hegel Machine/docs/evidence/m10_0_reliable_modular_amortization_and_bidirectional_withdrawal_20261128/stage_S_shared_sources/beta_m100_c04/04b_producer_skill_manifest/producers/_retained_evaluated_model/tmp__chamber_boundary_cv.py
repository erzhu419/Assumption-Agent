import importlib.util, pickle, numpy as np
from scipy import sparse
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ref=s['sensor_calibration_sweeps']['ref_rosette']; raw=s['sensor_calibration_sweeps']['raw_rosette']; D=np.c_[np.ones(len(ref)),ref]; err=np.empty_like(raw)
for i in range(len(ref)):
    keep=np.arange(len(ref))!=i; err[i]=raw[i]-D[i]@np.linalg.lstsq(D[keep],raw[keep],rcond=None)[0]
print('LOO calibration RMSE by channel',np.sqrt(np.mean(err**2,axis=0)))
y=b._calibrate(s); z=s['Z_nuisance']; ij=s['sensor_ij']; n=len(ij); shifts=len(s['mu_shifts']); wall=(ij[:,0]==0)|(ij[:,1]==0)|(ij[:,0]==s['nx']-1)|(ij[:,1]==s['ny']-1)
base=b._axis

def design(kind, sides):
    def axis(k,v,h,r):
        a=base(k,v,h,r).tolil(); idx=0 if len(k)==s['nx'] else 1
        for end, i, j, q in [(0,0,1,-1),(1,len(k)-1,len(k)-2,0)]:
            if not sides[2*idx+end]: continue
            if kind=='closed': a[i,i]-=r/h
            elif kind=='half': a[i,i]-=.5*r/h
            elif kind=='double': a[i,i]+=r/h
            elif kind=='strong': a[i,i]+=3*r/h
            elif kind=='reflected': a[i,i]+=(.5*(k[i]+k[j]))/h**2; a[i,j]-=(.5*(k[i]+k[j]))/h**2
            elif kind=='periodic':
                cross=.5*(k[i]+k[q])/h**2; a[i,i]+=cross-r/h; a[i,q]=-cross
            elif kind=='dirichlet': a[i,i]+=1e6
        return a.tocsc()
    b._axis=axis
    try: return b._forward(s)[0]
    finally: b._axis=base

def validate(X):
    errors=np.empty((shifts,n)); budget=s['total_budget']; cap=s['u_box']; E=np.eye(X.shape[1]); bank=np.broadcast_to(np.arange(n),(shifts,n)).ravel()
    for i in range(n):
        te=bank==i; tr=~te; Q=np.linalg.qr(z[tr])[0]; Xm=X[tr]-Q@(Q.T@X[tr]); ym=y[tr]-Q@(Q.T@y[tr]); w,_=b._box_simplex(Xm.T@Xm+1e-9*E,-Xm.T@ym,cap,budget); gam=np.linalg.lstsq(z[tr],y[tr]-X[tr]@w,rcond=None)[0]; errors[:,i]=y[te]-X[te]@w-z[te]@gam
    return errors
base_err=validate(b._forward(s)[0]); results=[]
for kind in ['closed','half','double','strong','reflected','periodic','dirichlet']:
    masks=[[1]*4,[1,1,0,0],[0,0,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] if kind!='periodic' else [[1]*4,[1,1,0,0],[0,0,1,1]]
    for mask in masks:
        err=validate(design(kind,mask)); d=np.mean(base_err**2-err**2,axis=0); results.append((np.mean(err**2)**.5,kind,mask,np.sqrt(np.mean(err[:,wall]**2)),np.sqrt(np.mean(err[:,~wall]**2)),np.mean(d)/(np.std(d,ddof=1)/np.sqrt(n))))
print('baseline pooled/wall/interior',*(np.sqrt(np.mean(x*x)) for x in (base_err,base_err[:,wall],base_err[:,~wall])))
print('baseline shift by wall and interior',np.sqrt(np.mean(base_err[:,wall]**2,axis=1)),np.sqrt(np.mean(base_err[:,~wall]**2,axis=1)))
for row in sorted(results)[:12]: print('challenger pooled/kind/mask/wall/interior/paired_z',row)

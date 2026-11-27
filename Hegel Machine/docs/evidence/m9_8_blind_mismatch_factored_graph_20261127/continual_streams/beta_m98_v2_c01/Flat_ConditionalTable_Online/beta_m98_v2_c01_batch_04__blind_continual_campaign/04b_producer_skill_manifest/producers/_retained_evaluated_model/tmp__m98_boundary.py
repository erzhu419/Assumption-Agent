import itertools, runpy, pickle
import numpy as np
from scipy.sparse import eye, kron, diags
from scipy.sparse.linalg import splu
p=runpy.run_path('/tmp/m98_prototype.py')
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
cal=st['sensor_calibration_sweeps']; x,y=cal['ref_rosette'],cal['raw_rosette']; good=np.max(abs(y),axis=1)<9
x,y=x[good],y[good]; keep=np.arange(len(x)); errors=[]
for i in keep:
  a=p['_fit_calibration'](x[keep!=i],y[keep!=i]); errors.append([p['_calibration_features'](x[i:i+1],j)[0]@a[j]-y[i,j] for j in range(4)])
print('compact calibration LOO RMSE by channel',np.mean(np.square(errors),axis=0)**.5,'max',np.max(np.abs(errors))); print('coefficients',p['_fit_calibration'](x,y))
arc='/tmp/hegel_continual_archive/beta_m98_v2_c01_batch_0'
axes=[runpy.run_path(arc+str(i)+'_solver.py')['_axis'] for i in [1,2,3]]
y=p['_calibrate'](st); z=st['Z_nuisance']; zr=st['nuisance_ridge']*np.eye(z.shape[1]); nx,ny=st['nx'],st['ny']; ns=len(st['sensor_ij']); nc=len(st['source_modes']); wr=st['source_ridge']*np.eye(nc)
rhs=np.stack([m.ravel(order='F') for m in st['source_modes']],axis=1); ii,jj=st['sensor_ij'].T; sensor=ii+nx*jj
wall=(ii==0)|(jj==0)|(ii==nx-1)|(jj==ny-1)
for ax,ay in itertools.product(range(3),repeat=2):
    mats=[]
    for j,k,n in [('x',ax,nx),('y',ay,ny)]:
        args=[st['kappa_'+j],st['vel_'+j],st['L'+j]/(n-1)]
        if k==1: args.append(st['robin_'+j])
        mats.append(axes[k](*args))
    L=kron(eye(ny),mats[0],format='csc')+kron(mats[1],eye(nx),format='csc')+diags(st['V_reaction_2d'].ravel(order='F'),format='csc')
    d=np.vstack([splu(L+mu*eye(nx*ny,format='csc')).solve(rhs)[sensor] for mu in st['mu_shifts']]); n=len(y)
    def fit(tr,te):
        D,Y,Z=d[tr],y[tr],z[tr]; gram=Z.T@Z+zr
        q=np.column_stack((D,Y)); q-=Z@np.linalg.solve(gram,Z.T@q); S,R=q[:,:-1],q[:,-1]
        w,nu=p['_box_simplex_qp'](S.T@S+wr,-S.T@R,st['u_box'],st['total_budget']); gamma=np.linalg.solve(gram,Z.T@(Y-D@w))
        obj=(np.linalg.norm(S@w-R)**2+st['source_ridge']*(w@w))/2
        return d[te]@w+z[te]@gamma-y[te],obj
    all_=np.ones(n,bool); _,obj=fit(all_,all_)
    e=np.empty(n)
    for j in range(ns):
        te=np.arange(n)%ns==j; e[te],_=fit(~te,te)
    e=e.reshape(-1,ns); rm=lambda x:np.mean(x*x)**.5
    es=[]
    for j in range(n//ns):
        te=np.arange(n)//ns==j; es.append(rm(fit(~te,te)[0]))
    print((ax,ay),'obj',round(obj,8),'LOSO',round(rm(e),8),'wall',round(rm(e[:,wall]),8),'worst sensor',round(max(rm(v) for v in e.T),8),'LO shift',np.round(es,8))

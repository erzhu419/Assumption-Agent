import importlib.util,itertools,json,pickle
import numpy as np
from scipy.sparse import eye,kron,diags
p='/tmp/hegel_continual_archive/dev_beta_m100_02_batch_04_solver.py'
spec=importlib.util.spec_from_file_location('prev',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; x=sw['ref_rosette']; y=sw['raw_rosette']; t=m._sweep_fit(s)
print('coefficients\n',t); e=np.abs(m._raw(x,t)-y); print('calibration abs error quantiles per output\n',np.quantile(e,[.5,.9,.95,.99,1.],axis=0)); obs=m._calibrate(s); print('calibrated range',obs.min(),obs.max(),'reference range',x.min(),x.max())
print('lagged telemetry: previous completed episode -> following chamber')
a=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for rec,name in zip(a,['s00','batch01','batch02','batch03','batch04','batch05']): print(rec['completed_campaign_short_name'],np.mean(rec['facility_telemetry_trace'],axis=0)[[0,4]],name)

def operator(s, modes):
    axes=[]
    for a,kind in zip('xy',modes):
        n=s['n'+a]; k=s['kappa_'+a]; v=s['vel_'+a]; h=s['L'+a]/(n-1)
        axes.append(m._periodic_axis(k,v,h) if kind=='periodic' else m._axis(k,v,h,s['robin_'+a] if kind=='open' else 0.))
    ax,ay=axes
    return kron(eye(s['ny']),ax,format='csc')+kron(ay,eye(s['nx']),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc')

def compare(s, obs):
    Z=s['Z_nuisance']; zr=s['nuisance_ridge']; sr=s['source_ridge']; I=np.eye(len(s['source_modes'])); IZ=np.eye(Z.shape[1]); n=len(obs)
    for modes in itertools.product(('open','sealed','periodic'),repeat=2):
        S,U=m._responses(s,operator(s,modes)); d=np.column_stack((S,obs)); errs=[]
        for test in [np.array([],dtype=int)]+[np.arange(i,i+4) for i in range(0,n,4)]:
            tr=np.ones(n,dtype=bool); tr[test]=False; z=Z[tr]; pp=np.linalg.solve(z.T@z+zr*IZ,z.T@d[tr]); st,yt=np.split(d[tr]-z@pp,[-1],axis=1); yt=yt.ravel()
            w,nu=m._box_simplex(st.T@st+sr*I,-st.T@yt,s['u_box'],s['total_budget']); gamma=pp[:,-1]-pp[:,:-1]@w
            if not test.size: train=np.sqrt(np.mean((S@w+Z@gamma-obs)**2)); weights=w
            else: errs.append(np.sqrt(np.mean((S[test]@w+Z[test]@gamma-obs[test])**2)))
        print(modes,'train/rmscv/worst',np.round([train,np.sqrt(np.mean(np.square(errs))),max(errs)],7),'w',np.round(weights,5))
if __name__=='__main__': compare(s,obs)

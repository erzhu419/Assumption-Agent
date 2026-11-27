import glob, importlib.util, inspect, pickle, json
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

base='/tmp/hegel_continual_archive/'
mods={}
for p in sorted(glob.glob(base+'*_solver.py')):
    key=p.split('_')[-2]
    spec=importlib.util.spec_from_file_location('m'+key,p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    mods[key]=m
    print(key, inspect.signature(m._axis), [k for k in vars(m) if 'rosette' in k])
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=item['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']
print('sweep max abs sorted',np.round(np.sort(np.max(abs(y),axis=1))[-15:],4))
print('single banks',[(k,type(v).__name__,getattr(v,'shape',None)) for k,v in c['single_bank_sweeps'].items()])
print('pulse weights',[p['w_pulse'].tolist() for p in item['campaign_diagnostics']])
good=np.max(abs(y),axis=1)<8
m=mods['06']; phi=m._rosette_basis(x)
aff=lambda a: np.column_stack([np.ones(len(a)),a])
for parity in [0,1]:
    tr=good & (np.arange(len(x))%2==parity); te=good & ~tr
    aa=np.linalg.lstsq(aff(x[tr]),y[tr],rcond=None)[0]
    cc=m._fit_rosette({'ref_rosette':x[tr],'raw_rosette':y[tr]})
    print('sweep fold',parity,'affine/nonlinear raw RMSE',np.sqrt(np.mean((aff(x[te])@aa-y[te])**2)),np.sqrt(np.mean((np.einsum('njk,jk->nj',phi[te],cc)-y[te])**2)))
cc=m._fit_rosette(c); aa=np.linalg.lstsq(aff(x[good]),y[good],rcond=None)[0]
print('rosette coefficients',np.round(cc,6))
Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; pp=item['campaign_diagnostics']
for p in pp:
    p['affine']=np.linalg.solve(aa[1:].T,(p['y_pulse_raw'].reshape(-1,4)-aa[0]).T).T.ravel()
    p['nonlinear']=m._invert_rosette(p['y_pulse_raw'].reshape(-1,4),cc).ravel()

def matrices(s,rx,ry):
    nx,ny=s['nx'],s['ny']; A=[]
    for name,n,reg in [('x',nx,rx),('y',ny,ry)]:
        a=mods[reg]._axis; args=(s['kappa_'+name],s['vel_'+name],s['L'+name]/(n-1))
        A.append(a(*args,s['robin_'+name]) if len(inspect.signature(a).parameters)==4 else a(*args))
    L=kron(eye(ny),A[0],format='csc')+kron(A[1],eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    rhs=np.stack([a.ravel(order='F') for a in s['source_modes']],axis=1); i,j=s['sensor_ij'].T
    D=[splu(L+mu*eye(nx*ny,format='csc')).solve(rhs) for mu in s['mu_shifts']]
    return np.vstack([a[i+nx*j] for a in D]), D
for rx in ['01','02','03','06']:
    for ry in ['01','02','03','06']:
        S,_=matrices(s,rx,ry)
        vals=[]
        for p in pp:
            z=S@p['w_pulse']; raw=p['y_pulse_raw'];
            rr=[z-p['affine'],z-p['nonlinear'], (aff(z.reshape(-1,4))@aa).ravel()-raw,np.einsum('njk,jk->nj',m._rosette_basis(z.reshape(-1,4)),cc).ravel()-raw]
            vals.append([round(np.linalg.norm(v)/np.sqrt(v.size),6) for v in rr]+[round(np.linalg.norm(rr[1]-Q@(Q.T@rr[1]))/np.sqrt(z.size),6)])
        print(rx,ry,vals)
log=json.load(open(base+'archive_apparatus_log.json')); traces=np.asarray([r['facility_telemetry_trace'] for r in log]); print('trace shape',traces.shape); print('latest trace distances',np.round(np.linalg.norm((traces-traces[-1]).reshape(len(traces),-1),axis=1),4))

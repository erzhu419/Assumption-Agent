import importlib.util, pickle, numpy as np
from itertools import combinations_with_replacement
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c=st['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; b=c['single_bank_sweeps']; obs=st['y_obs_raw'].reshape(-1,4)
terms=[()]+[t for d in range(1,4) for t in combinations_with_replacement(range(4),d)]
def design(x,terms): return np.column_stack([np.prod(x[:,t],axis=1) for t in terms])
mask=np.max(np.abs(y),axis=1)<5
P=design(x,terms); fit=np.linalg.lstsq(P[mask],y[mask],rcond=None)[0]; r=y-P@fit
print('clean count, rank, max error',mask.sum(),np.linalg.matrix_rank(P[mask]),abs(r[mask]).max()); print('excluded rosettes: ref | residual',np.c_[x[~mask],r[~mask]].round(5)); print('significant terms',[(t,v.round(5).tolist()) for t,v in zip(terms,fit) if abs(v).max()>0.001]);
keep=abs(fit).max(axis=1)>0.001; terms=[t for t,k in zip(terms,keep) if k]; P=P[:,keep]; fit=np.linalg.lstsq(P[mask],y[mask],rcond=None)[0]
er=[]
for i in np.flatnonzero(mask):
 m=mask.copy(); m[i]=False; er.append(P[i]@np.linalg.lstsq(P[m],y[m],rcond=None)[0]-y[i])
print('clean rosette leave-one-out max',np.max(abs(np.array(er)),axis=0)); bx=np.eye(4)[b['bank_id']]*b['ref_signal'][:,None]; br=b['raw_readout']-(design(bx,terms)@fit)[np.arange(len(bx)),b['bank_id']]; print('single-bank residual by bank',br.reshape(4,-1).round(5))
from scipy.optimize import root
inv=[root(lambda v: (design(v[None,:],terms)@fit)[0]-yy,np.zeros(4),tol=1e-11) for yy in obs]; cal=np.array([r.x for r in inv]).ravel(); print('observation inverse max residual',max(abs(r.fun).max() for r in inv))
z=st['Z_nuisance']; p=np.linalg.solve(z.T@z+st['nuisance_ridge']*np.eye(z.shape[1]),z.T); ry=cal-z@(p@cal)
for i in ('s00','batch_01','batch_02','batch_03'):
 spec=importlib.util.spec_from_file_location('m','/tmp/hegel_continual_archive/dev_beta_m98_00_'+i+'_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); s,u=m._design(st); a=s-z@(p@s); ridge=st['source_ridge']; w,nu=m._box_qp(a.T@a+ridge*np.eye(a.shape[1]),-a.T@ry,st['u_box'],st['total_budget']); r=a@w-ry; print(i,'objective',0.5*(r@r+ridge*(w@w)),'rmse/shifts',np.sqrt(np.mean(r.reshape(len(st['mu_shifts']),-1)**2,axis=1)), 'weights', w.round(4))

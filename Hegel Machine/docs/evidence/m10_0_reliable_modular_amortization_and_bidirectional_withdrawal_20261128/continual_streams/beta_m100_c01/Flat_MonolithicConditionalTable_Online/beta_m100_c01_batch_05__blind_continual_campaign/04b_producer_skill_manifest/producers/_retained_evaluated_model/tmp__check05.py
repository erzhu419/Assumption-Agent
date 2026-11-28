import importlib.util, pickle, numpy as np
from itertools import product
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; x,y=sw['ref_rosette'],sw['raw_rosette']; good=np.max(abs(y),axis=1)<5; banks=sw['single_bank_sweeps']; row=np.arange(len(x))
p=m._fit_calibration(x,y,banks); print('calibration',p.round(8),'corrupt rosettes',row[~good]); print('calibration clean max',np.max(abs(m._response(x[good],p)-y[good]))); print('calibration CV', [np.sqrt(np.mean((m._response(x[good&(row%5==i)],m._fit_calibration(x[row%5!=i],y[row%5!=i],banks))-y[good&(row%5==i)])**2)) for i in range(5)])
y=m._calibrate(s); z=s['Z_nuisance']; groups=np.arange(len(y)).reshape(len(s['mu_shifts']),-1,4); rows=np.arange(len(y)); ij=s['sensor_ij']; edge=(ij[:,0]==0)|(ij[:,0]==s['nx']-1)|(ij[:,1]==0)|(ij[:,1]==s['ny']-1)
def score(S, test, source_ridge, nuisance_ridge, full=False):
 tr=rows if full else np.setdiff1d(rows,np.ravel(test)); ts=np.ravel(test); Z=z[tr]; P=np.linalg.solve(Z.T@Z+nuisance_ridge*np.eye(Z.shape[1]),Z.T); A=S[tr]-Z@(P@S[tr]); r=y[tr]-Z@(P@y[tr]); H=A.T@A+source_ridge*np.eye(S.shape[1]); w,_=m._box_simplex(H,-A.T@r,s['u_box'],s['total_budget']); return np.sqrt(np.mean((S[ts]@w+z[ts]@(P@(y[tr]-S[tr]@w))-y[ts])**2))
for walls in map(''.join,product('RNP',repeat=2)):
 S,_=m._transport(s,walls)
 print(walls, [[round(v,8) for v in [score(S,rows,a,b,True),max(score(S,t,a,b) for t in groups),max(score(S,t,a,b) for t in groups.transpose(1,0,2)),score(S,rows[np.tile(edge,len(groups))],a,b)]] for a,b in [(s['source_ridge'],s['nuisance_ridge']),(1e-9,1e-9)]])

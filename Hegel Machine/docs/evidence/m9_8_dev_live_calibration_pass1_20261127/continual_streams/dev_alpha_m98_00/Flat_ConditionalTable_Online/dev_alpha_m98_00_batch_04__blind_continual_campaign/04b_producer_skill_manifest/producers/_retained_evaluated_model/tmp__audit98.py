import pickle, runpy, numpy as np
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
m=runpy.run_path('/holdout_task/batch_solver.py'); globals().update({k:v for k,v in m.items() if k.startswith('_') and callable(v)})
C=D['calibration_load_cells']; U,R=C['true_channel_loads'],C['bridge_raw']; fit=_calibration(C); ordinary=fit[-1]; err=R-fit[2](U)@fit[0]
print('calibration excluded true/raw/residual',np.round(np.c_[U,R,err][~ordinary],5)); print('residual norms ordinary max vs excluded',np.linalg.norm(err[ordinary],axis=1).max(),np.linalg.norm(err[~ordinary],axis=1)); print('excluded total residual',err[~ordinary].sum(1))
# Check if omitted cubic or rational/pair-difference terms transfer to real held-out calibration slices.
a=fit[2](U)
full=np.column_stack([U[:,I].prod(1) for k in (2,3) for I in combinations_with_replacement(range(4),k)])
pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
rat=np.column_stack([z for i,j in pairs for q in (U[:,i]-U[:,j],U[:,i]+U[:,j]) for z in (q/(1+.25*q*q),q/(1+q*q))])
axis=np.count_nonzero(U,axis=1)==1
folds=np.array_split(np.random.default_rng(12).permutation(np.flatnonzero(ordinary)),8)
for name,A in [('sparse',a),('full3',np.c_[a,full]),('rational',np.c_[a,rat])]:
 e=np.full_like(R,np.nan)
 for ii in folds:
  train=ordinary.copy(); train[ii]=False
  e[ii]=A[ii]@np.linalg.lstsq(A[train],R[train],rcond=None)[0]-R[ii]
 print(name,'heldout axial-forward rmse ordinary/single-axis/multiaxis and max',*[np.sqrt(np.mean(e[g].sum(1)**2)) for g in (ordinary,ordinary&axis,ordinary&~axis)],np.max(np.abs(e[ordinary].sum(1))))
tr,te,pi=(D[k] for k in ('train','test','chamber_verification_trials')); dt,t=D['past_dt'],D['eval_times']; xtr,xte=_features(tr,te); a=_inputs(tr,xtr,fit); xp,=_features(pi,balanced=False); p=_inputs(pi,xp,fit); tau,th,s=_material(tr,a,dt,t)
for b,z in [(tr,a),(pi,p)]:
 y,r=_predict(z,dt,t,tau,th); e=b['strain_curves_obs']-b['strain_0'][:,None]-y; er=b['instant_rate_obs']-r
 print('lab, count, curve median, rate median, curve RMSE, rate RMSE',[(int(j),int(g.sum()),round(np.median(e[g]),6),round(np.median(er[g]),6),round(np.sqrt(np.mean(e[g]**2)),6),round(np.sqrt(np.mean(er[g]**2)),6)) for j in np.unique(b['lab_id']) for g in [b['lab_id']==j]])
print('spectrum',tau,'weights',th)

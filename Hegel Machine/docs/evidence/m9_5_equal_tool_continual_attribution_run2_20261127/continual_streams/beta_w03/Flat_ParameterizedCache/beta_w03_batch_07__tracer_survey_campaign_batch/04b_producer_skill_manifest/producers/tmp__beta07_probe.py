import pickle, pathlib, numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares
p=pathlib.Path('/tmp/hegel_continual_archive/beta_w03_batch_06_solver.py')
ns={}; exec(p.read_text().replace('1.45 * np.asarray', 'np.asarray'), ns)
e=pickle.load(open('/holdout_task/public_beta_w03_batch_07.pkl','rb'))[0]
s,gt=e['study'],e['gt']; response=ns['_responses'](s)
i,j=s['sensor_ij'].T
D=response[:,:,i,j].transpose(0,2,1).reshape(-1,len(s['source_modes']))
Z=s['Z_nuisance']; U,d,_=np.linalg.svd(Z,full_matrices=False); P=np.eye(len(Z))-(U*(d*d/(d*d+s['nuisance_ridge'])))@U.T
print('profiled design error',np.max(abs(P@D-gt['S_tilde'])), 'terminal error', np.linalg.norm(np.einsum('m,mxy->xy',gt['w_opt'],response[-1])-gt['terminal_field_A']))
r=s['row_bank_id']; y=s['y_obs_raw']; sw=s['sensor_calibration_sweeps']; pars=[]
for b in range(4):
 k=sw['bank_id']==b; pars.append(np.polyfit(sw['ref_signal'][k],sw['raw_readout'][k],1))
clean=r<2; g,o=np.array(pars).T; yc=(y-o[r])/g[r]; A=np.column_stack([D,Z]); coeff=lstsq(A[clean],yc[clean])[0]; pred=A@coeff
print('clean fit rms',np.std(yc[clean]-pred[clean]),'clean rank',np.linalg.matrix_rank(A[clean]),'clean weight sum',coeff[:D.shape[1]].sum())
for b in range(2,4):
 k=r==b; f=sw['bank_id']==b; x=sw['ref_signal'][f]; z=sw['raw_readout'][f]; g0,o0=np.polyfit(pred[k],y[k],1)
 print('bank',b,'clean-wing prediction fit',g0,o0,'spot residuals',z-(g0*x+o0))
 for x0,z0 in zip(x,z):
  slope=(pred[k]-x0)@(y[k]-z0)/np.sum((pred[k]-x0)**2); print('anchor',x0,'gain offset',slope,z0-slope*x0,'RMS',np.sqrt(np.mean((y[k]-z0-slope*(pred[k]-x0))**2)))

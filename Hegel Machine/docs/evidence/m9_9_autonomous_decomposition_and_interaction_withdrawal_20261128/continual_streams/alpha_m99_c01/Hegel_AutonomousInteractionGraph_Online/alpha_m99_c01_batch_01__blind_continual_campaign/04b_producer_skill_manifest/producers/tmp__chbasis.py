exec(open('/tmp/chprobe.py').read().split('for name,g,xx')[0])
from scipy.optimize import least_squares
from scipy.linalg import lstsq
Y=np.column_stack((di['strain_curves_obs']-di['strain_0'][:,None],di['instant_rate_obs']))
def basis(tau):
 return np.vstack((np.column_stack((-np.expm1(-t[:,None]/tau),t)),np.r_[1/tau,1]))
def profile(tau):
 A=basis(tau); C=lstsq(A,Y.T)[0].T; return C,Y-C@A.T
C,E=profile(tau); print('fixed spectrum residual quantiles by output',np.quantile(abs(E[:,:-1]),[.5,.9,1]),np.quantile(abs(E[:,-1]),[.5,.9,1]))
f=least_squares(lambda u:profile(np.exp(u))[1].ravel(),np.log(tau)); T=np.exp(f.x); C,E=profile(T); print('free chamber spectrum',T,'resid quantiles',np.quantile(abs(E),[.5,.9,1])); print('Y tail singular values',np.linalg.svd(Y,compute_uv=False))
a,rr=m._design(T,xd,pd,hd,dt,t); gap=rr[:,:2]*T; drift=rr[:,2:]@beta[2:]; D=C-np.column_stack((gap*beta[:2],drift)); print('amplitudes',np.round(C[:10],5)); print('corrections',np.round(D[:10],5))
F=np.column_stack((hd,pd,gap,xd,drift,di['lab_id'],di['stratum'])); names=['hold']+[f'p{k}' for k in range(pd.shape[1])]+['g0','g1']+[f'x{k}' for k in range(xd.shape[1])]+['drift','lab','stratum'];
for v in (C,D):
 print('correlations',[(n,*np.round([np.corrcoef(F[:,j],v[:,k])[0,1] for k in range(3)],3)) for j,n in enumerate(names)])
for name,s in [('train',tr),('test',te),('diag',di)]:
 print(name,'sum of loads by channel mean/sd',[(np.round(np.mean(s[k],axis=tuple(range(s[k].ndim-1))),3),np.round(np.std(s[k],axis=tuple(range(s[k].ndim-1))),3)) for k in ['past_channel_raw','hold_channel_raw']])
cal=d['calibration_load_cells']; raw=cal['bridge_raw']; y=cal['true_channel_loads']; a=np.c_[np.ones(len(raw)),raw]; print('cal affine component errors RMS',np.sqrt(np.mean((a@lstsq(a,y)[0]-y)**2,axis=0))); print('sweeps',[(l,np.round(lstsq(np.c_[np.ones(sum(cal['single_channel_sweeps']['lab_id']==l)),cal['single_channel_sweeps']['cmd_stress'][cal['single_channel_sweeps']['lab_id']==l]],cal['single_channel_sweeps']['true_stress'][cal['single_channel_sweeps']['lab_id']==l])[0],5)) for l in np.unique(cal['single_channel_sweeps']['lab_id'])]);

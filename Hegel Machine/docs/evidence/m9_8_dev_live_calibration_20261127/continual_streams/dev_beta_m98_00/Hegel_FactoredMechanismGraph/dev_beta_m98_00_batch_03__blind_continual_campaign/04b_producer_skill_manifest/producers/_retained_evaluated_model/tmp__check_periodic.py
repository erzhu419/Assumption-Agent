from fit_wrap import s, W, Y, design
import numpy as np
for leakage in (0,1e-6):
    p=[]
    for a in 'xy':
        k=s['kappa_'+a]; v=s['vel_'+a]; h=s['L'+a]/(len(k)-1)
        p.extend([(0.5*(k[0]+k[1])+max(-v[0],0)*h)/k[0], (0.5*(k[-1]+k[-2])+max(v[-1],0)*h)/k[-1], leakage, leakage, (0.5*(k[0]+k[-1])+max(v[0],0)*h)/k[0], (0.5*(k[0]+k[-1])+max(-v[-1],0)*h)/k[-1]])
    e=(design(p)@W-Y).reshape(len(s['mu_shifts']),-1,2)
    print('leakage',leakage,'coefficients',p,'shift/pulse RMSE',np.sqrt(np.mean(e**2,axis=1)), 'worst bank RMS',np.max(np.sqrt(np.mean(e.reshape(-1,4,2)**2,axis=0))))

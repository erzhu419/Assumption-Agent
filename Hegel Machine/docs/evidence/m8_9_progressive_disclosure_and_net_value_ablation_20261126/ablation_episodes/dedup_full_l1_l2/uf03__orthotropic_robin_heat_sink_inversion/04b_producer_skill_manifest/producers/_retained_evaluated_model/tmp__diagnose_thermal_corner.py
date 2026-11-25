import runpy
from unittest.mock import patch
from scipy.optimize import least_squares
import numpy as np

# Reuse the independent forward generator without running its test loops.
path = '/tmp/stress_thermal.py'
text = open(path).read()
setup = text[:text.index('for i, c in enumerate(cases):')]
generator = text[text.index('def synthetic('):text.index('for group,n,gamma,sigma,xy,contrast,mcount in [')]
ns = {}
exec(compile(setup + generator, path, 'exec'), ns)
mod = ns['mod']
for seed in range(8):
    m, gt = ns['synthetic'](22,2.2,.085,(.03,.96),601+seed,.7,52)
    x = (np.arange(m['N'])+.5)/m['N']
    g = mod._sensor_green(m['kappa'], m['gamma_robin'], m['sensor_ij'])
    def shape(xy):
        z = np.exp(-.5*((x[:,None]-xy[0])**2+(x[None,:]-xy[1])**2)/m['sigma_m']**2)
        return np.einsum('mij,ij->m',g,z)
    p0 = np.r_[gt['source_xy'],gt['amplitude'],gt['junction_offset']]
    clean = p0[2]*shape(p0[:2])+p0[3]
    y = m['T_obs']
    ok = np.abs(y-clean)<.02
    def predict(p):
        return p[2]*shape(p[:2])+p[3]
    fit = least_squares(lambda p:(predict(p)-y)[ok],p0,bounds=([0,0,0,-np.inf],[1,1,np.inf,np.inf]),gtol=1e-13)
    sd = .00125*np.sqrt(np.linalg.inv(fit.jac.T@fit.jac)[2,2])/p0[2]
    alt = []
    for factor in [.84,1.18]:
        amp = p0[2]*factor
        def profile(xy):
            r = (amp*shape(xy)-clean)[ok]
            return r-r.mean()
        f = least_squares(profile,p0[:2],bounds=(0,1),gtol=1e-14)
        alt.append((np.round(f.x,5).tolist(), round(np.linalg.norm(f.fun)/.00125,3)))
    ans = mod.localize_thermal_source(m)
    print(seed,'oracle signed amplitude error %',round(100*(fit.x[2]/p0[2]-1),3),'solver',round(100*(ans['amplitude']/p0[2]-1),3),'local relative sd %',round(100*sd,3),'disjoint-tolerance alternatives (xy, separation in noise standard deviations)',alt)

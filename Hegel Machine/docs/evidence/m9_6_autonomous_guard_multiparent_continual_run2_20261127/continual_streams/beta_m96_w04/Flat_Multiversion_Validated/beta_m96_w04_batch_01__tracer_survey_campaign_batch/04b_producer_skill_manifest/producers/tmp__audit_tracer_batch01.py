import importlib.util
import pickle
import numpy as np

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', p+'beta_m96_w04_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for item in pickle.load(open(p+'public_beta_m96_w04_batch_01.pkl','rb')):
    s, gt = item['study'], item['gt']
    for a in 'xy':
        k, v, reaction = (np.asarray(s[name+'_'+a]) for name in ('kappa','vel','reaction'))
        L, n, rg = s['L'+a], len(k), s['robin_'+a]
        h = L/n
        op = m._build_1d(L,n,k,v,rg,reaction)
        q = np.random.default_rng(27).normal(size=n)
        ref = reaction*q
        for i in range(n-1):
            flux = 2/(1/k[i]+1/k[i+1])*(q[i]-q[i+1])/h**2
            ref[i] += flux
            ref[i+1] -= flux
        for i in range(n):
            upstream = i-1 if v[i] >= 0 else i+1
            ref[i] += abs(v[i])*(q[i]-(q[upstream] if 0<=upstream<n else 0))/h
        for i in (0,n-1):
            ref[i] += q[i]/(h/rg+h*h/(2*k[i]))
        err = np.max(abs(op@q-ref))
        print('stencil', a, 'max error',err)
        assert np.allclose(op@q,ref,rtol=1e-11,atol=1e-10)
    sw=s['sensor_calibration_sweeps']
    banks=s['row_bank_id']
    loo=[]
    for b in np.unique(banks):
        x,z=(np.asarray(sw[k])[sw['bank_id']==b] for k in ('ref_signal','raw_readout'))
        res=[]
        for i in range(len(x)):
            keep=np.arange(len(x))!=i
            g,c=m._bank_line(x[keep],z[keep])
            res.append(abs(g*x[i]+c-z[i]))
        loo.append(max(res))
    print('leave-one-reference-out maximum residual by bank',loo)
    assert max(loo)<1e-7
    S,U=m._forward(s)
    y=m._calibrate(s)
    design,response=m._profile(s,S,y)
    for b in np.unique(banks):
        keep=banks==b
        print('bank',b,'profiled design/response max errors',np.max(abs(design[keep]-gt['S_tilde'][keep])),np.max(abs(response[keep]-gt['r_y'][keep])))
        assert np.max(abs(design[keep]-gt['S_tilde'][keep]))<1e-10
        assert np.max(abs(response[keep]-gt['r_y'][keep]))<1e-7

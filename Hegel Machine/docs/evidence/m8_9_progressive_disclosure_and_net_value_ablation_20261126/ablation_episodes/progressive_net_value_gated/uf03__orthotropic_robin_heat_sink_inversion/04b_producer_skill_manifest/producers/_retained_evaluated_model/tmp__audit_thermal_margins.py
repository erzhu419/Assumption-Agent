import importlib.util
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

path = Path('/holdout_task/thermal_sink_solver.py')
spec = importlib.util.spec_from_file_location(path.stem, path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with path.with_name('public_thermal_cases.pkl').open('rb') as f:
    cases = pickle.load(f)

for i, case in enumerate(cases):
    m, gt = case['measurement'], case['gt']
    n, k = m['N'], m['kappa']
    sigma, gamma = m['sigma_m'], m['gamma_robin']
    ij = m['sensor_ij']
    ids = n*ij[:, 0]+ij[:, 1]
    data = m['T_obs']
    x, y = (np.indices((n, n))+.5)/n
    # Independently enumerate all four faces: corner cells have two sinks.
    matrix = lil_matrix((n*n, n*n))
    for a, b in np.ndindex(n, n):
        row = a*n+b
        for c, d in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
            if 0 <= c < n and 0 <= d < n:
                w = 2/(1/k[a,b]+1/k[c,d])
                matrix[row, row] += w
                matrix[row, n*c+d] -= w
            else:
                matrix[row, row] += gamma/n
    independent = splu(matrix.tocsc())
    package = solver._factor_operator(k, gamma)
    def basis(xy):
        dx, dy = x-xy[0], y-xy[1]
        src = np.exp(-.5*(dx*dx+dy*dy)/sigma**2)/n**2
        return np.stack((src, src*dx/sigma**2, src*dy/sigma**2), axis=-1).reshape(n*n,3)

    q = basis(gt['source_xy'])
    states = independent.solve(q)
    assert np.max(np.abs(states-package.solve(q))) < 1e-12
    flux_total = gamma/n*(states[:,0].reshape(n,n)[[0,-1],:].sum()+states[:,0].reshape(n,n)[:,[0,-1]].sum())
    assert abs(q[:,0].sum()-flux_total) < 1e-12
    fit = solver.localize_thermal_source(m)
    a, b = fit['amplitude'], fit['junction_offset']
    f, fx, fy = independent.solve(basis(fit['source_xy']))[ids].T
    r = a*f+b-data
    med = np.median(r)
    mad = 1.4826*np.median(abs(r-med))
    good = abs(r-med) <= 6*mad
    assert good.sum() > 4
    rc = r[good]-r[good].mean()
    d = r[good, None]-r[None, good]
    assert np.isclose(np.sum(rc**2), np.sum(d*d)/(2*good.sum()), rtol=1e-12, atol=1e-15)
    amp_contrast = np.dot(f[good]-f[good].mean(),data[good]-data[good].mean())/np.var(f[good])/good.sum()
    jac = np.column_stack((a*fx,a*fy,a*f,np.ones_like(f)))[good]
    sd = np.sqrt(np.diag(np.linalg.inv(jac.T@jac))) * np.linalg.norm(r[good])/np.sqrt(good.sum()-4)
    errs = np.array([np.linalg.norm(np.subtract(fit['source_xy'],gt['source_xy'])),abs(a/gt['amplitude']-1),abs(b-gt['junction_offset'])])
    print('public', i, 'excluded', int((~good).sum()), 'inlier MAD scale', mad, 'contrast-only relative amp delta', abs(amp_contrast/a-1), 'local parameter SDs (x, y, log A, offset)', sd, 'fraction of tolerance', errs/[.035,.08,.008])
    assert np.all(errs <= [.035,.08,.008])

    # Same real plate and sensor layout, but a synthetic source pair close to
    # a wall. Quantify separability for the noise used in the prior edge stress.
    q1 = independent.solve(basis((.006,.74))[:,0])[ids]
    q2 = independent.solve(basis((.025,.74))[:,0])[ids]
    keep = np.argsort(q1)[:-round(.06*len(ids))]
    q1, q2 = q1[keep], q2[keep]
    a1 = 24.
    a2 = a1*np.dot(q1-q1.mean(), q2-q2.mean())/np.sum((q2-q2.mean())**2)
    delta = a1*q1-a2*q2
    separation = np.linalg.norm(delta-delta.mean())/.0015
    print('layout edge pair', i, 'amplitudes', (a1,a2), '8%-tolerance intervals disjoint', max(a1,a2)*.92 > min(a1,a2)*1.08, 'joint separation / noise SD', separation)

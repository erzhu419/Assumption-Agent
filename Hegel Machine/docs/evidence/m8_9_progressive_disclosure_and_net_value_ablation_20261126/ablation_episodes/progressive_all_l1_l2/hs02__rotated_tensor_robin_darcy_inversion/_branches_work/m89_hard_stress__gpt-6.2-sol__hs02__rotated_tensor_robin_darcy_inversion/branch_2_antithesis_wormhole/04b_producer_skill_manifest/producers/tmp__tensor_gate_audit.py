import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from tensor_darcy_robin import infer_tensor_darcy_sources, _operator


def face_balance(p, u):
    xx, xy, yy = (p[k] for k in ('k_xx', 'k_xy', 'k_yy'))
    nx, ny = u.shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    gamma, ambient = p['gamma_robin'], p['u_ambient']
    out = hx*hy*p['sigma']*u
    for i in range(nx):
        for j in range(ny):
            if i+1 < nx:
                lo, hi = max(j-1, 0), min(j+1, ny-1)
                flux = (hy/hx)*2/(1/xx[i,j]+1/xx[i+1,j])*(u[i,j]-u[i+1,j])
                flux -= (xy[i,j]+xy[i+1,j])/8*(u[i,hi]-u[i,lo]+u[i+1,hi]-u[i+1,lo])
                out[i,j] += flux
                out[i+1,j] -= flux
            if j+1 < ny:
                lo, hi = max(i-1, 0), min(i+1, nx-1)
                flux = (hx/hy)*2/(1/yy[i,j]+1/yy[i,j+1])*(u[i,j]-u[i,j+1])
                flux -= (xy[i,j]+xy[i,j+1])/8*(u[hi,j]-u[lo,j]+u[hi,j+1]-u[lo,j+1])
                out[i,j] += flux
                out[i,j+1] -= flux
            out[i,j] += ((int(i==0)+int(i==nx-1))*hy/(1/gamma+hx/(2*xx[i,j])) + (int(j==0)+int(j==ny-1))*hx/(1/gamma+hy/(2*yy[i,j])))*(u[i,j]-ambient)
    return out


with open('/holdout_task/public_tensor_cases.pkl', 'rb') as f:
    cases = pickle.load(f)['cases']
rng = np.random.default_rng(461)
for k, case in enumerate(cases):
    p, g = case['problem'], case['gt']
    u0, w0, o0 = (g[key] for key in ('u_true', 'source_weights', 'sensor_offset'))
    grid = np.asarray(p['sensor_ij'])
    clean = np.abs(p['y_obs']-u0[tuple(grid.T)]-o0)<0.1
    cropped = {**p, **{key: p[key][:13,:19].copy() for key in ('k_xx','k_xy','k_yy','sigma')}}
    cropped['k_xy'] *= -1
    for q in (p, cropped):
        A, b, _ = _operator(q)
        u = rng.normal(size=q['k_xx'].shape)
        actual = (A@u.ravel()-b).reshape(u.shape)
        expected = face_balance(q, u)
        mismatch = np.max(np.abs(actual-expected))
        assert mismatch < 1e-11, mismatch
        print(k, u.shape, 'independent flux discrepancy', mismatch)
    fit = infer_tensor_darcy_sources(p)
    shifted = infer_tensor_darcy_sources({**p, 'y_obs':p['y_obs']+2.25})
    delta_w = np.max(abs(shifted['source_weights']-fit['source_weights']))
    delta_u = np.max(abs(shifted['predicted_field']-fit['predicted_field']))
    delta_o = abs(shifted['sensor_offset']-fit['sensor_offset']-2.25)
    assert max(delta_w, delta_u, delta_o) < 1e-10
    print(k, 'bias equivariance max error', max(delta_w, delta_u, delta_o))
    spatial = []
    mid = (np.array(u0.shape)-1)/2
    for sx, sy in ((0,0),(0,1),(1,0),(1,1)):
        test = ((grid[:,0]>mid[0])==sx)&((grid[:,1]>mid[1])==sy)
        ii, jj = grid[test & clean].T
        res = infer_tensor_darcy_sources({**p,'sensor_ij':grid[~test],'y_obs':p['y_obs'][~test]})
        w, o, u = (res[key] for key in ('source_weights','sensor_offset','predicted_field'))
        e = p['y_obs'][test & clean]-u[ii,jj]-o
        spatial.append((np.linalg.norm(w-w0)/np.linalg.norm(w0),abs(o-o0),np.max(abs(u-u0)),np.max(abs(e))))
    print(k, 'worst spatial holdout: relative source, offset, field max, clean held-out residual max:', np.max(spatial, axis=0))

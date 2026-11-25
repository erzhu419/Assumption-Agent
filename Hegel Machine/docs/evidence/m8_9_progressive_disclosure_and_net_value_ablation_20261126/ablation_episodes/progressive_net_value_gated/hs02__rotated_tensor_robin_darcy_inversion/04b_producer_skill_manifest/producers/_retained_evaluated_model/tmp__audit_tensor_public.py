import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('tensor_darcy_robin', '/holdout_task/tensor_darcy_robin.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_tensor_cases.pkl', 'rb') as stream:
    cases = pickle.load(stream)['cases']

for index, case in enumerate(cases):
    p, g = case['problem'], case['gt']
    ij = p['sensor_ij']
    shape = p['k_xx'].shape
    flat = np.ravel_multi_index(ij.T, shape)
    response = mod._forward(p)
    result = mod.infer_tensor_darcy_sources(p)
    u, s, off = (result[key] for key in ('predicted_field', 'source_weights', 'sensor_offset'))
    hx, hy = p['Lx'] / shape[0], p['Ly'] / shape[1]
    gamma, ambient = p['gamma_robin'], p['u_ambient']
    kx, ky = p['k_xx'], p['k_yy']
    x, y = kx[[0,-1], :], ky[:, [0,-1]]
    balance = (hy * gamma * x / (x + gamma * hx / 2) * (u[[0,-1], :] - ambient)).sum()
    balance += (hx * gamma * y / (y + gamma * hy / 2) * (u[:, [0,-1]] - ambient)).sum()
    balance += hx * hy * np.sum(p['sigma'] * u - np.einsum('m,mij->ij', s, p['basis']))
    assert abs(balance) < 1e-11
    shift = 7.5
    transpose = dict(p, Lx=p['Ly'], Ly=p['Lx'], k_xx=ky.T, k_yy=kx.T, k_xy=p['k_xy'].T,
                     sigma=p['sigma'].T, basis=p['basis'].transpose(0,2,1), sensor_ij=ij[:, ::-1], y_obs=p['y_obs']+shift)
    reflected = dict(p, k_xx=kx[::-1], k_yy=ky[::-1], k_xy=-p['k_xy'][::-1], sigma=p['sigma'][::-1],
                     basis=p['basis'][:, ::-1], sensor_ij=np.column_stack((shape[0]-1-ij[:,0], ij[:,1])))
    for transformed, mapped, moved_offset in ((transpose, u.T, off+shift), (reflected, u[::-1], off)):
        r = mod.infer_tensor_darcy_sources(transformed)
        np.testing.assert_allclose(r['source_weights'], s, rtol=0, atol=1e-9)
        np.testing.assert_allclose(r['predicted_field'], mapped, rtol=0, atol=1e-10)
        np.testing.assert_allclose(r['sensor_offset'], moved_offset, rtol=0, atol=1e-10)
    true_noise = p['y_obs'] - g['u_true'][tuple(ij.T)] - g['sensor_offset']
    clean = np.abs(true_noise) < .75
    d = response[flat[clean], 1:]
    d_centered = d - d.mean(axis=0)
    _, sv, vt = np.linalg.svd(d_centered, full_matrices=False)
    # Standard-error multiplier for the field mean; the common sensor offset is profiled out.
    mean_gain = np.linalg.norm((response[:, 1:].mean(axis=0) @ vt.T) / sv)
    worst = np.zeros(3)
    wrong_masks = 0
    for j in range(len(ij)):
        keep = np.arange(len(ij)) != j
        r = mod.infer_tensor_darcy_sources(dict(p, sensor_ij=ij[keep], y_obs=p['y_obs'][keep]))
        resid = p['y_obs'] - r['predicted_field'][tuple(ij.T)] - r['sensor_offset']
        wrong_masks += int((abs(resid[j]) < .75) != clean[j])
        worst = np.maximum(worst, [np.linalg.norm(r['source_weights']-g['source_weights'])/np.linalg.norm(g['source_weights']), abs(r['sensor_offset']-g['sensor_offset']), np.linalg.norm(r['predicted_field']-g['u_true'])/np.linalg.norm(g['u_true'])])
    print(f'public {index}: conservation {abs(balance):.3g}; transpose+shift and reflection pass; centered singular range [{sv[-1]:.4g}, {sv[0]:.4g}]; field-mean noise gain {mean_gain:.4g}; held-out spike-mask errors {wrong_masks}/{len(ij)}; worst delete-one source/offset/field {worst}')

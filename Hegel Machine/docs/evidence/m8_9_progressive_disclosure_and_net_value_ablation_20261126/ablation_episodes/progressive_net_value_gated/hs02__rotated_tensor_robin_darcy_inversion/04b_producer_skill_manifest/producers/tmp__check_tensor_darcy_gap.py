import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('tensor_darcy_robin', '/tmp/tensor_darcy_gap.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_tensor_cases.pkl', 'rb') as stream:
    cases = pickle.load(stream)['cases']
rng = np.random.default_rng(445)

def errors(p, weights, offset, field):
    result = mod.infer_tensor_darcy_sources(p)
    assert set(result) == {'source_weights', 'sensor_offset', 'predicted_field'}
    assert result['source_weights'].shape == weights.shape
    assert result['predicted_field'].shape == field.shape
    assert type(result['sensor_offset']) is float
    return (np.linalg.norm(result['source_weights']-weights)/np.linalg.norm(weights),
            abs(result['sensor_offset']-offset),
            np.linalg.norm(result['predicted_field']-field)/np.linalg.norm(field))

def direct_balance(p, u):
    kxx,kxy,kyy,sigma = [p[k] for k in ('k_xx','k_xy','k_yy','sigma')]
    nx,ny = u.shape
    hx,hy = p['Lx']/nx,p['Ly']/ny
    gamma,ambient = p['gamma_robin'],p['u_ambient']
    w = np.pad(u, 1, mode='edge')
    cx = w[2:,1:-1]-w[:-2,1:-1]
    cy = w[1:-1,2:]-w[1:-1,:-2]
    out = hx*hy*sigma*u
    fx = hy/hx*(2*kxx[:-1]*kxx[1:]/(kxx[:-1]+kxx[1:]))*(u[:-1]-u[1:]) - (kxy[:-1]+kxy[1:])*(cy[:-1]+cy[1:])/8
    fy = hx/hy*(2*kyy[:,:-1]*kyy[:,1:]/(kyy[:,:-1]+kyy[:,1:]))*(u[:,:-1]-u[:,1:]) - (kxy[:,:-1]+kxy[:,1:])*(cx[:,:-1]+cx[:,1:])/8
    out[:-1] += fx
    out[1:] -= fx
    out[:,:-1] += fy
    out[:,1:] -= fy
    for i in (0,-1):
        out[i] += hy*gamma*kxx[i]/(kxx[i]+gamma*hx/2)*(u[i]-ambient)
    for j in (0,-1):
        out[:,j] += hx*gamma*kyy[:,j]/(kyy[:,j]+gamma*hy/2)*(u[:,j]-ambient)
    return out

worst_balance = 0.
for k,c in enumerate(cases):
    p,g = c['problem'],c['gt']
    print('public', k, 'source_rel offset_abs field_rel', errors(p,g['source_weights'],g['sensor_offset'],g['u_true']))
    R = mod._forward(p)
    ij = p['sensor_ij']
    clean = abs(p['y_obs']-g['u_true'][tuple(ij.T)]-g['sensor_offset']) < .05
    held_clean = []
    for j in np.where(clean)[0]:
        keep = np.arange(len(ij)) != j
        result = mod.infer_tensor_darcy_sources(dict(p,sensor_ij=ij[keep],y_obs=p['y_obs'][keep]))
        held_clean.append(abs(result['predicted_field'][tuple(ij[j])]+result['sensor_offset']-p['y_obs'][j]))
    print('public leave-one-clean-sensor-out maximum absolute error', max(held_clean))

base = cases[0]['problem']
for label,change in [('public_geometry',{}), ('negative_cross',{'k_xy': -base['k_xy']}), ('weak_robin_rectangular', {'Lx': 1.85, 'Ly': .65, 'gamma_robin': .16}), ('strong_robin',{'gamma_robin': 6.0})]:
    p = dict(base,**change)
    R = mod._forward(p)
    shape = p['k_xx'].shape
    test_u = rng.normal(size=shape)
    A,b,_ = mod._assemble(p)
    wrong = np.max(abs((A @ test_u.ravel()-b).reshape(shape)-direct_balance(p,test_u)))
    worst_balance = max(worst_balance, wrong)
    np.testing.assert_allclose((A @ test_u.ravel()-b).reshape(shape),direct_balance(p,test_u),atol=1e-12)
    flat = np.ravel_multi_index(p['sensor_ij'].T,shape)
    x = np.column_stack((R[flat,1:],np.ones(len(flat))))
    leverage = (np.linalg.qr(x)[0]**2).sum(axis=1)
    slices = {}
    for mode in ('noiseless','random_spikes','high_leverage_spikes'):
        values = []
        for j in range(24):
            weights = rng.uniform(.5,3,len(p['basis']))
            if j % 2:
                weights *= rng.choice([-1,1],len(weights))
            offset = rng.uniform(-3,3)
            field = (R[:,0] + R[:,1:] @ weights).reshape(shape)
            observations = field.ravel()[flat]+offset
            if mode != 'noiseless':
                observations += rng.normal(0,.0015,len(flat))
                out = rng.choice(len(flat),6,replace=False) if mode == 'random_spikes' else np.argsort(leverage)[-6:]
                observations[out] += rng.uniform(1.5,4,len(out))*rng.choice([-1,1],len(out))
            values.append(errors(dict(p,y_obs=observations),weights,offset,field))
        slices[mode] = np.round(np.max(values,axis=0),7).tolist()
    print(label, slices)
print('stress stencil max absolute balance difference',worst_balance)

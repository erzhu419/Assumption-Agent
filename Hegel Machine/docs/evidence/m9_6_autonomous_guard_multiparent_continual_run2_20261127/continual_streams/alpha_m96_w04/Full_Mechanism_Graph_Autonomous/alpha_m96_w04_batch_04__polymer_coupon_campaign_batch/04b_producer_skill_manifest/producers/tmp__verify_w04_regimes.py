exec(open('/tmp/validate_w04_slices.py').read().split('coef =')[0])
lab, cmd, obs = m._checks(d['calibration_load_cells'])
gain, bias = map(np.asarray, (g['stress_gains'], g['stress_biases']))
truth = np.column_stack((gain, bias))
for wing in (0, 1):
    for offset in range(2):
        sparse, spiked = 2*wing+offset, 2*(1-wing)+offset
        keep = lab != sparse
        keep[np.flatnonzero(lab == sparse)[0]] = True
        q = obs.copy()
        i = np.flatnonzero(lab == spiked)[np.argsort(cmd[lab == spiked])[-2:]]
        q[i] += np.array([.7, -.9])
        coef = m._calibration(dict(lab_id=lab[keep], cmd_stress=cmd[keep], true_stress=q[keep]))
        err = np.max(np.abs(coef-truth))
        print('sparse lab, spiked lab, max coefficient error:', sparse, spiked, err)
        assert err < .001

t, dt, sy = d['eval_times'], d['past_dt'], d['yield_stress']
x = m._aligned((tr, te))
n = len(tr['lab_id'])
tau = np.asarray(g['taus']); w = np.asarray(g['theta']); vp = np.asarray(g['vp_theta'])

def make(part, features, past_limit, hold_limit):
    part = dict(part)
    c = part['lab_id']
    past = np.clip(part['past_stress_cmd']*gain[c,None]+bias[c,None], -past_limit, past_limit)
    hold = np.clip(part['stress_hold_cmd']*gain[c]+bias[c], -hold_limit, hold_limit)
    part.update(past_stress_cmd=(past-bias[c,None])/gain[c,None], stress_hold_cmd=(hold-bias[c])/gain[c])
    o = np.sign(hold)*np.maximum(abs(hold)-sy,0)**2
    damage = np.sum(np.maximum(abs(past)-sy,0)**2*dt,axis=1)*hold
    slope = w[2]+features@w[3:]+vp[0]*o
    curve = part['strain_0'][:,None]+slope[:,None]*t
    rate = slope.copy()
    for j in range(2):
        state = np.zeros(len(c))
        for step, delta in enumerate(dt):
            a = np.exp(-delta/tau[j]); state = a*state+(1-a)*past[:,step]
        amp = w[j]*(hold-state)
        if j == 0: amp += vp[1]*damage
        curve += amp[:,None]*(1-np.exp(-t/tau[j]))
        rate += amp/tau[j]
    part['instant_rate_obs'] = rate
    return part, curve, rate

for name, p, h, active in [('sub-yield', .85*sy, .85*sy, [False,False]), ('history-only', np.inf, .85*sy, [False,True]), ('hold-only', .85*sy, np.inf, [True,False])]:
    a, a_curve, _ = make(tr,x[:n],p,h); a['strain_curves_obs'] = a_curve
    b, curve, rate = make(te,x[n:],p,h)
    b.pop('instant_rate_obs')
    exact_checks = dict(lab_id=lab, cmd_stress=cmd, true_stress=gain[lab]*cmd+bias[lab])
    out = solver({**d, 'train':a, 'test':b, 'calibration_load_cells':exact_checks})
    error = [np.max(np.abs(out['predicted_strain']-curve)), np.max(np.abs(out['initial_rate_pred']-rate))]
    weights = out['viscoplastic_weights']
    print(name, 'max curve/rate error', error, 'plastic weights', weights)
    assert np.all(weights[~np.array(active)] == 0) and np.allclose(weights[active], vp[active], atol=.001)
    assert max(error) < .005

exec(open('/tmp/audit_final.py').read().split('results=[]')[0])
max_change = np.zeros(2)
base = m.solve_batch(s)
for panel in ('panel_a', 'panel_b'):
    p = s[panel]
    for j in range(len(p['u'])):
        q = {k: np.delete(v, j, axis=0) for k,v in p.items()}
        out = m.solve_batch(dict(s, **{panel:q}))
        for k, name in enumerate(('y_hat', 'state_pred')):
            max_change[k] = max(max_change[k], np.max(np.abs(out[name] - base[name])))
print('Max public test prediction change after deleting one calibration/operator row:', max_change)
assert np.all(max_change < 0.001)
assert all(np.isfinite(v).all() for v in base.values())
y, last = base['y_hat'], base['state_pred'][:, -1]
np.testing.assert_allclose(base['diagnostics_vec'], [y.mean(), y.std(), last.mean(), last.std()])

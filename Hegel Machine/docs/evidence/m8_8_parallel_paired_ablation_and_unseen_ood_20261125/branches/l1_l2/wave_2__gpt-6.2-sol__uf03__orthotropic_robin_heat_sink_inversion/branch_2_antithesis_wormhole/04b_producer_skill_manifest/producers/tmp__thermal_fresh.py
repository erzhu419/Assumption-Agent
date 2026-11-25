from pathlib import Path
src = Path('/tmp/thermal_identifiability.py').read_text()
exec(src[:src.index('for label,center')])
for label in ('interior', 'edge', 'corner'):
    results = []
    for seed in range(36):
        rng = np.random.default_rng(70170+seed)
        axis = rng.uniform(.025,.15)
        xy = {'interior':rng.uniform(.16,.84,2), 'edge':np.array([axis,rng.uniform(.16,.84)]), 'corner':np.array([axis,rng.uniform(.025,.15)])}[label]
        if seed%2: xy = xy[::-1]
        if seed%4 >= 2: xy[seed%2] = 1-xy[seed%2]
        flavor = seed%3
        d,g,bad,clean = simulate(22,52,2.2,.085,.6,xy,77400+seed,.0013,'moderate' if flavor == 1 else 'gross')
        if flavor == 2:
            d['T_obs'][bad] = np.quantile(clean,.25)
        results.append((seed,run(d,g)))
    report('fresh '+label, results)

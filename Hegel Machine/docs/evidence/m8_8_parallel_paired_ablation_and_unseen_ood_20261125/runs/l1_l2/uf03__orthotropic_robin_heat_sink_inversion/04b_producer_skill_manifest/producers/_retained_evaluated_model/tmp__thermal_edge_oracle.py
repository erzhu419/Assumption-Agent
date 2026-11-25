from pathlib import Path
s = Path('/tmp/thermal_oracle.py').read_text()
exec(s[:s.index('for c in pickle.load')])
exec(s[s.index('def compare'):s.index('part=src')])
s = Path('/tmp/thermal_identifiability.py').read_text()
exec(s[s.index('exec(src[src.index'):s.index("for label,center")])
for corruption in ('gross', 'moderate'):
    results = []
    for seed in range(48):
        rng = np.random.default_rng(19020+seed)
        xy = np.array([rng.uniform(.02,.16),rng.uniform(.16,.84)])
        d,g,bad,_ = simulate(22,52,2.2,.085,.6,xy,29020+seed,.0013,corruption)
        r,masks,sd,norm = compare(d,g,bad)
        results.append(r)
        if np.any(r[0]>LIMIT) or np.any(r[1]>LIMIT):
            print(corruption,seed,'actual/oracle/oracle-from-actual',np.round(r,5).tolist(),'gross kept / clean removed',masks,'oracle linearized standard errors',np.round(sd,5).tolist())
    a = np.asarray(results)
    print(corruption,'misses by fit and metric',np.sum(a>LIMIT,axis=0).tolist(),'maxima',a.max(axis=0).tolist())

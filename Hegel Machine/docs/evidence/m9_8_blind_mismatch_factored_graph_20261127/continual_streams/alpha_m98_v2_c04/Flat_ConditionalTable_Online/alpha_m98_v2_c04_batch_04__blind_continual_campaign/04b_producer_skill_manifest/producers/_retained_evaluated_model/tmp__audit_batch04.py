import importlib.util, json, pickle
import numpy as np

s = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
c = d['calibration_load_cells']
x, y = c['true_channel_loads'], c['bridge_raw']
# Challenge the cyclic forward law with omitted quadratic/cubic interactions
# and bounded pairwise terms. Freeze all choices before evaluating each fold.
pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
triples = [(i,j,k) for i in range(4) for j in range(i+1,4) for k in range(j+1,4)]
prev = (np.arange(4)-1)%4

def terms(z, mode):
    base = m._bridge_basis(z)
    if mode == 'cyclic':
        return base
    if mode == 'cubic':
        return np.column_stack([base, z[:,0]*z[:,2], z[:,1]*z[:,3],
            *[z[:,i]**2*z[:,j] for i in range(4) for j in range(4) if i != j],
            *[z[:,i]*z[:,j]*z[:,k] for i,j,k in triples]])
    return np.column_stack([base, *[(z[:,i]*z[:,j])/(1+a*(z[:,i]+sgn*z[:,j])**2)
        for i,j in pairs for sgn in (-1,1) for a in (0.5,)]])

results = {mode:[[],[]] for mode in ('cyclic','cubic','rational')}
for grouping in (np.arange(len(x))%5, np.argsort(np.argsort(np.linalg.norm(x,axis=1)))//16):
    for g in np.unique(grouping):
        train = grouping != g
        k, clean = m._fit_bridge(x[train], y[train])
        # Separate sum and decontaminated channel scores: the sum uses every held-out
        # observation, while channel scores exclude large within-row discrepancies.
        e0 = y[~train] - terms(x[~train], 'cyclic')@k
        clean_test = np.linalg.norm(e0,axis=1) < .002
        for mode,(sums,channels) in results.items():
            a = terms(x[train][clean],mode)
            coef = np.linalg.lstsq(a,y[train][clean],rcond=None)[0]
            e = terms(x[~train],mode)@coef-y[~train]
            sums.extend(abs(e.sum(1)))
            channels.extend(np.max(abs(e[clean_test]),axis=1))
for mode,(sums,channels) in results.items():
    print(mode, 'all-row raw-sum p90/max',np.quantile(sums,.9),max(sums), 'clean-row channel max',max(channels))

log = json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
traces = np.asarray([row['facility_telemetry_trace'] for row in log]); means = traces.mean(1)
for omit in range(traces.shape[2]):
    distance = np.linalg.norm(np.delete(means[:-1]-means[-1],omit,axis=1),axis=1)
    j = np.argsort(distance)
    print('omit apparatus channel',omit,'nearest archive row',j[0],'runner-up gap',distance[j[1]]-distance[j[0]])

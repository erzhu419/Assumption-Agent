import pickle
import numpy as np

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
cal = d['calibration_load_cells']
u = cal['bridge_raw']
y = cal['true_channel_loads'].sum(axis=1)
a = np.c_[np.ones(len(u)), u]
q = np.column_stack([*(u[:, j] for j in range(4)), *(u[:, i] + sign*u[:, j] for i in range(4) for j in range(i+1,4) for sign in (-1,1))])**2
fold = np.random.default_rng(98).permutation(len(u)) % 5
p = np.empty((3, len(u)))

def predict(tr, te, j=None, rational=False):
    if j is None:
        return a[te] @ np.linalg.lstsq(a[tr], y[tr], rcond=None)[0]
    term = q[:, j]
    if rational:
        b = np.linalg.lstsq(np.c_[a[tr], -y[tr]*term[tr]], y[tr], rcond=None)[0]
        den = 1 + b[-1]*term[te]
        return a[te] @ b[:-1] / den if np.min(den) > .5 else np.full(np.sum(te), np.inf)
    g = np.c_[a, term]
    return g[te] @ np.linalg.lstsq(g[tr], y[tr], rcond=None)[0]

for k in range(5):
    te = fold == k
    p[0, te] = predict(~te, te)
    for m, rational in enumerate((False, True), 1):
        scores = [sum(np.sum((predict((fold != k) & (fold != i), fold == i, j, rational)-y[fold == i])**2) for i in range(5) if i != k) for j in range(q.shape[1])]
        j = int(np.argmin(scores))
        p[m, te] = predict(~te, te, j, rational)
        print('outer fold/model/selected term', k, m, j)
for name, pred in zip(('affine', 'pairwise square', 'rational'), p):
    e = abs(pred-y)
    rms = [np.sqrt(np.mean(e[fold == k]**2)) for k in range(5)]
    print(name, 'pooled RMS', np.sqrt(np.mean(e**2)), 'worst fold RMS', max(rms), 'high-load p90', np.quantile(e[y >= np.quantile(y, .75)], .9))

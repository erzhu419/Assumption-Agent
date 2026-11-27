import importlib.util,pickle,numpy as np
P='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',P+'alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open(P+'public_alpha_m97_c03_batch_04.pkl','rb'))[0]['dataset']; tr=d['train']; lab=tr['lab_id']; st=tr['stratum']; n=len(lab)
p,h,x,o,dt,t=m._prepare(d); D=m._damage(d,p,dt)[:n]; p,h,x=p[:n],h[:n],x[:n]
y=np.column_stack((tr['strain_curves_obs']-o[:n,None], tr['instant_rate_obs']))
def args(i): return p[i],h[i],x[i],dt,t,D[i]
def metrics(e):
 return np.array([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.median(np.abs(e[:,:-1])),np.median(np.abs(e[:,-1])),np.max(np.abs(e[:,:-1])),np.max(np.abs(e[:,-1]))])
print('dataset fields:', sorted(d), 'train fields:', sorted(tr))
print('calibration rows by lab:', np.unique(d['calibration_load_cells']['lab_id'], return_counts=True))
def fit_args(i):
    return p[i], h[i], x[i], dt, t, D[i], True

def extra(i):
    # Stress-normalized, pairwise-feature, and three-way interaction challengers.
    a = np.tanh(D[i]); q = h[i]; f = x[i]; logt = np.r_[np.log1p(t), 1.0]
    cols = [q/(1+q*q), a*q/(1+q*q), a*q*(f[:,0]-f[:,1]), a*q*(f[:,2]+f[:,3])]
    cols.extend(f[:,j]*f[:,k] for j in range(4) for k in range(j+1,4))
    cols.extend(a*q*f[:,j] for j in range(4))
    return np.column_stack(cols)[:,None,:]*logt[None,:,None]

for name, groups in [('lab', lab), ('stratum', st)]:
    worst = np.zeros((2,2)); shifts = []
    for k in np.unique(groups):
        v = groups == k
        pp, ww = m._fit(fit_args(~v), y[~v])
        e0 = y[~v] - m._basis(pp, *fit_args(~v))@ww
        e = y[v] - m._basis(pp, *fit_args(v))@ww
        sc = np.r_[np.repeat(m._scale(e0[:,:-1]), len(t)), m._scale(e0[:,-1])]
        weight = 1 / (sc * np.hypot(1, e0/(2.385*sc)))
        coef = np.linalg.lstsq((extra(~v)*weight[:,:,None]).reshape(-1,14), (e0*weight).ravel(), rcond=None)[0]
        e2 = e - extra(v)@coef
        for act in (False, True):
            q = (D[v]>0)==act
            if q.any():
                row = [[np.median(np.abs(r[q,:-1])), np.median(np.abs(r[q,-1]))] for r in (e,e2)]
                worst = np.maximum(worst, row)
                shifts.append([k, int(act), np.median(e[q,:-1]), np.median(e[q,-1])])
    print(name, 'worst strain/rate median AE, baseline then additions:', worst.tolist())
    print(name, 'held-out signed medians [group, active, strain, rate]:', np.round(shifts,5).tolist())

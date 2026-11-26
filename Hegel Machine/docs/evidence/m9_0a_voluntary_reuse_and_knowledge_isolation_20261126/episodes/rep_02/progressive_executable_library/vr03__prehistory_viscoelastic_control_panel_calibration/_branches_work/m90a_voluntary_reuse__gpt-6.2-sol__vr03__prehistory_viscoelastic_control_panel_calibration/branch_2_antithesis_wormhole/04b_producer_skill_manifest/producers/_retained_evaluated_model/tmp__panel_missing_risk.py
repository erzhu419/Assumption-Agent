import pickle, sys
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/holdout_task')
import viscoelastic_panel_calibration as m

nodes, weights = np.polynomial.hermite.hermgauss(20)
weights /= np.sqrt(np.pi)

def compare(x, s, c, y, xq, sq):
    w = m.balanced_cell_weights(c, s)
    fill = m.weighted_column_fill(x, w)
    u = m._features(np.where(np.isnan(x), fill, x), s)
    uq = m._features(np.where(np.isnan(xq), fill, xq), sq)
    baseline = LogisticRegression(C=5, max_iter=500, tol=1e-9).fit(u, y, sample_weight=w)
    complete = np.isfinite(x).all(axis=1)
    wc = w[complete] / w[complete].mean()
    fit = LogisticRegression(C=5, max_iter=500, tol=1e-9).fit(u[complete], y[complete], sample_weight=wc)
    cov = np.cov(x[complete], rowvar=False, aweights=wc, ddof=0)
    # Preserve the specified mean imputation; integrate unobserved logit variation around it.
    missing = np.isnan(xq)
    beta_missing = missing * fit.coef_[0, :3]
    sigma = np.sqrt(np.einsum('ni,ij,nj->n', beta_missing,cov,beta_missing))
    mu = fit.decision_function(uq)
    return np.column_stack([baseline.predict_proba(uq)[:,1], expit(mu[:,None] + np.sqrt(2)*sigma[:,None]*nodes) @ weights])

splits = {'LOCO': [], 'public query': []}
for i, case in enumerate(pickle.load(open('/holdout_task/public_viscoelastic_cohort.pkl','rb'))):
    data, gt = case['cohort'],case['gt']; tr, q = data['train'],data['query']
    x,xq = [m._aligned(split,data['taus']) for split in (tr,q)]
    c,s,y = [tr[key] for key in ('center_id','stratum','y')]
    for domain, centers, truth, z, pred in [('LOCO',c,y,x,None), ('public query',q['center_id'],gt['y_query'],xq,compare(x,s,c,y,xq,q['stratum']))]:
        for j in np.unique(centers):
            k = centers == j
            p = compare(x[~k],s[~k],c[~k],y[~k],x[k],s[k]) if pred is None else pred[k]
            yy = truth[k]; miss = np.isnan(z[k]).any(axis=1)
            metrics = [(roc_auc_score(yy, col), np.mean((yy-col)**2), np.mean((yy[miss]-col[miss])**2)) for col in p.T]
            splits[domain].append(metrics)
            print(domain,i,int(j),int(miss.sum()),np.round(metrics,5).tolist())
for name, metrics in splits.items():
    a = np.array(metrics)
    print(name,'min AUC',a[:,:,0].min(0),'max Brier',a[:,:,1].max(0),'missing mean Brier',a[:,:,2].mean(0))

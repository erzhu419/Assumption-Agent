import importlib.util
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location('calibration', '/holdout_task/viscoelastic_panel_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def features(z, s, basis):
    pair = np.column_stack([z[:, i]*z[:, j] for i, j in ((0,1),(0,2),(1,2))])
    extra = {'linear': (), 'quadratic': (z*z, pair),
             'rational': (z/(1+0.25*z*z), pair/(1+0.25*np.sum(z*z,axis=1,keepdims=True)))}[basis]
    return np.column_stack((z, np.eye(3)[s], *extra))

def predict(z, c, s, y, q, sq, basis, C):
    w = m._cell_weights(c, s)
    f = np.nansum(z*w[:,None],axis=0)/np.sum(np.isfinite(z)*w[:,None],axis=0)
    a = features(np.where(np.isfinite(z), z, f), s, basis)
    b = features(np.where(np.isfinite(q), q, f), sq, basis)
    return LogisticRegression(C=C,max_iter=500,tol=1e-8).fit(a,y,sample_weight=w).predict_proba(b)[:,1]

def scores(y,p):
    sensitivity = np.mean(p[y==1]>=.5)
    specificity = np.mean(p[y==0]<.5)
    return np.array([roc_auc_score(y,p), np.mean((y-p)**2), .5*(sensitivity+specificity), sensitivity, specificity])

for k, item in enumerate(pickle.load(open('/holdout_task/public_viscoelastic_cohort.pkl','rb'))):
    co, gt = item['cohort'], item['gt']
    tr, qu = co['train'], co['query']
    taus = np.asarray(co['taus'])
    z, q = (m._aligned_moduli(p,taus) for p in (tr,qu))
    raw, rawq = (np.where(p['missing_channel_mask'], np.nan, np.array([m._trace_moduli(t,taus) for t in p['traces']])) for p in (tr,qu))
    c,s,y = (tr[a] for a in ('center_id','stratum','y'))
    cq,sq,yq = qu['center_id'],qu['stratum'],gt['y_query']
    def loco(z, c, s, y, cfg):
        return np.array([scores(y[c==j],predict(z[c!=j],c[c!=j],s[c!=j],y[c!=j],z[c==j],s[c==j],*cfg)) for j in np.unique(c)])
    for label, a, b in [('raw',raw,rawq), ('aligned',z,q)]:
        p = predict(a,c,s,y,b,sq,'linear',10.)
        print(k,label,'pooled',scores(yq,p).round(4),'query center', [scores(yq[cq==j],p[cq==j]).round(4).tolist() for j in np.unique(cq)])
    cfgs = [(basis,C) for basis in ('linear','quadratic','rational') for C in (0.3,1.,3.,10.,30.)]
    for basis in ('linear','quadratic','rational'):
        cfg = [x for x in cfgs if x[0]==basis]
        outer, choices = [], []
        for j in np.unique(c):
            a = c!=j
            def key(x):
                h = loco(z[a],c[a],s[a],y[a],x)
                return (np.min(h[:,0]), -np.max(h[:,1]))
            best = max(cfg,key=key)
            choices.append(best[1]); outer.append(scores(y[~a],predict(z[a],c[a],s[a],y[a],z[~a],s[~a],*best)))
        outer = np.asarray(outer)
        print(k,basis,'nested C',choices,'worst AUC, Brier, BA, sensitivity, specificity', np.array([outer[:,0].min(),outer[:,1].max(),*outer[:,2:].min(axis=0)]).round(4))
    for C in (0.3,1.,3.,10.,30.):
        h = loco(z,c,s,y,('linear',C))
        p = predict(z,c,s,y,q,sq,'linear',C)
        g = np.array([scores(yq[cq==j],p[cq==j]) for j in np.unique(cq)])
        print(k,'linear',C,'LOCO + query worst AUC/Brier',*[round(v,5) for v in (h[:,0].min(),h[:,1].max(),g[:,0].min(),g[:,1].max())])

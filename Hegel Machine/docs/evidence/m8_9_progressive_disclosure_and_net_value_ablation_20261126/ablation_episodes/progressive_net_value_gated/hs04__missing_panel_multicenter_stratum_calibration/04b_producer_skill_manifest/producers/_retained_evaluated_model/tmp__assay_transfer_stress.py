import importlib.util
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

spec = importlib.util.spec_from_file_location('assay', '/holdout_task/multicenter_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
settings = [(1., 1.), (.3, 0.), (.3, .25), (.3, .5), (.3, 1.)]

for center_bias, biomarker_spread in [(0., .03), (1.8, .03), (0., .25), (1.8, .25)]:
    results = {key: [] for key in settings}
    oracle, maes = [], []
    for seed in range(30):
        rng = np.random.default_rng(seed + 14000)
        d = 8
        shift = np.array([3.5, 3.5, -3.5, -3.5, -2.8, 2.9])[:, None] + biomarker_spread*rng.normal(size=(6,d))
        scale = np.array([1.35, .85, 1.25, .8, 1.45, .75])[:, None] * np.exp(biomarker_spread*rng.normal(size=(6,d)))
        rc = np.repeat(np.arange(6), [80, 80, 80, 80, 65, 45])
        ref = shift[rc] + scale[rc]*rng.normal(size=(len(rc),d))
        ref[rng.random(ref.shape) < .15] = np.nan
        sparse = rng.choice(d, size=2, replace=False)
        rr = np.flatnonzero(rc == 5)
        ref[rr, sparse[0]] = np.nan
        ref[rr[1:], sparse[1]] = np.nan
        beta = rng.normal(size=d)
        beta *= 2.3 / np.linalg.norm(beta)
        parts, latent, probabilities = [], [], []
        for centers, n in [(np.arange(4),95),(np.arange(4,6),150)]:
            c = np.repeat(centers, n)
            s = rng.choice(3, len(c), p=[.55,.3,.15])
            z = rng.normal(size=(len(c),d)) + np.array([.45,.45,-.45,-.45,.45,-.45])[c,None]
            x = shift[c] + scale[c]*z
            x[rng.random(x.shape) < .035] = np.nan
            p = expit(np.array([-.8,.0,.8])[s] + z@beta + center_bias*np.array([1.,1.,-1.,-1.,0.,0.])[c])
            parts.append(dict(center_id=c, stratum=s, X=x, y=(rng.random(len(c)) < p).astype(int)))
            latent.append(z)
            probabilities.append(p)
        t,q = parts
        trans = m._reference_transforms(dict(center_id=rc,X=ref))
        xt,xq = (m._align(p,*trans) for p in parts)
        st,sq = (np.eye(3)[p['stratum']] for p in parts)
        ct = np.eye(4)[t['center_id']]; ct -= ct.mean(0)
        for center in (4,5):
            mask = q['center_id'] == center
            maes.append(abs(xq[mask]-latent[1][mask]).mean())
            y = q['y'][mask]
            oracle.append((roc_auc_score(y,probabilities[1][mask]),brier_score_loss(y,probabilities[1][mask])))
        for C,a in settings:
            model = LogisticRegression(C=C,max_iter=1000).fit(np.c_[xt,st,a*ct],t['y'])
            pred = model.predict_proba(np.c_[xq,sq,np.zeros((len(xq),4))])[:,1]
            for center in (4,5):
                mask = q['center_id'] == center
                results[C,a].append((roc_auc_score(q['y'][mask],pred[mask]),brier_score_loss(q['y'][mask],pred[mask])))
    print('center bias, scanner heterogeneity:',center_bias,biomarker_spread,'alignment median/max',np.round([np.median(maes),max(maes)],3),'oracle mean',np.round(np.mean(oracle,0),3))
    for key,z in results.items():
        z=np.asarray(z)
        print(key,'mean',np.round(z.mean(0),3),'AUROC p10/Brier p90',np.round([np.quantile(z[:,0],.1),np.quantile(z[:,1],.9)],3))

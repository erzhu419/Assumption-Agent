import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/b03_validate.py').read())
from scipy.optimize import least_squares

def robust(model, y, mask, kind):
    valid = mask & np.isfinite(y)
    t, c = s._fit(model, y, mask)
    residual = model.design(t) @ c - y
    sigma = []
    for j in range(y.shape[1]):
        v = residual[:, j][valid[:, j]]
        sigma.append(max(1.4826 * np.median(abs(v - np.median(v))), 1e-6) if len(v) else 1.)
    sigma = np.asarray(sigma)
    obs = y[valid]
    base = 1 / np.broadcast_to(sigma, y.shape)[valid]
    x = np.log(t)
    pred = model.design(t) @ c
    for _ in range(12):
        u = (pred[valid]-obs)*base
        w = 1 / np.sqrt(1 + (u / 2.5)**2) if kind == 'cauchy' else np.minimum(1, np.sqrt(1.5 / np.maximum(np.abs(u), 1e-10)))
        scale = base * w
        target = obs * scale
        def evaluate(v):
            D = model.design(np.exp(v))[valid] * scale[:, None]
            b = np.linalg.lstsq(D, target, rcond=None)[0]
            return np.einsum('ij,j->i',D,b)-target,b
        opt = least_squares(lambda v: evaluate(v)[0], x, bounds=(np.log([0.03, 0.05]), np.log([50., 150.])),max_nfev=40)
        b = evaluate(opt.x)[1]
        newpred = model.design(np.exp(opt.x)) @ b
        if np.max(abs(newpred-pred)) < 1e-7:
            return np.exp(opt.x), b
        x,pred = opt.x,newpred
    return np.exp(x), b

print('absolute baseline residual quantiles by output', np.nanquantile(abs(full),[0.50,0.80,0.90,0.95,0.99],axis=0).round(4), sep='\n')
for kind in ['baseline','huber','cauchy']:
    fit = s._fit if kind == 'baseline' else lambda m,y,mask:robust(m,y,mask,kind)
    t,c=fit(m,y,np.ones_like(y,dtype=bool)); print(kind,'full tau/coef',t.round(5),c.round(5))
    for lab in np.unique(a['lab_id']):
        held=a['lab_id']==lab
        t,c=fit(m,y,np.broadcast_to(~held[:,None],y.shape)); e=(m.design(t)@c-y)[held]
        print(kind,int(lab),'tau',t.round(5),'curve median/p80/MAE',np.nanquantile(abs(e[:,:-1]),[.50,.80]).round(5),round(np.nanmean(abs(e[:,:-1])),5),'rate RMSE',round(rmse(e[:,-1]),5))

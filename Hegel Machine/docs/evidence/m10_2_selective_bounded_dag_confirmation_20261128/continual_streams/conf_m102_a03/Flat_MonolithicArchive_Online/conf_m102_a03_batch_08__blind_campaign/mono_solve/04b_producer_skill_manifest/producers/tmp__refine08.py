exec(open('/tmp/finalcv08.py').read().split('E=[]')[0])
from pathlib import Path
from scipy.optimize import minimize_scalar
path=Path('/holdout_task/batch_solver.py'); orig=path.read_text()
y=t['y']; base,Q=m._features(z,w); d=z[:,3]

def evaluate(opts):
    cols, penalty, exp, equal = opts
    out=[]
    for i,j in S:
        cyc=base[:,5:]; avg=cyc.sum(1)
        def prof(q, predict=False):
            den=1+q*d*d
            A=np.c_[base[:,:5]/den[:,None],avg,cyc[:,:3]-cyc[:,3:],Q[:,cols],d/den,d**3/den**exp]
            scale=np.sqrt((A[i]**2).mean(0)).clip(1e-8); U=A/scale; B=U[i]
            reg=np.r_[np.zeros(6),np.full(3,equal),np.full(len(cols),penalty),0.,penalty]
            c=np.linalg.solve(B.T@B+np.diag(reg+1e-10),B.T@y[i]); e=B@c-y[i]
            return U[j]@c if predict else e@e+np.dot(reg,c*c)
        q=minimize_scalar(prof,bounds=(.10,.7),method='bounded').x
        out.append(np.abs(prof(q,True)-y[j]).mean())
    return np.array(out)
from itertools import product
options=list(product([tuple(range(10)),(1,2,5,6),(2,4,6),(2,6),(1,2,5,6,9)], [.0003,.004,.03,.12],[1.,1.5,2.],[.001,.1,1.])); rows=[]
for opts in options:
    e=evaluate(opts)
    if e[0]<=.019 and max(e[:10])<=.024:
        rows.append((max(e[10:])+0.35*max(e[:10]),opts,e))
rows.sort(key=lambda row:row[0]); print('challengers',[(opts,e.round(5).tolist()) for _,opts,e in rows[:5]])
if rows and max(rows[0][2][10:])<.09:
    _,(cols,penalty,exp,equal),_=rows[0]
    new='''def _scalar(z, w, nt, y):
    base, quad = _features(z, w)
    d = z[:, 3]
    cyclic = base[:, 5:]
    total = cyclic.sum(axis=1)
    contrasts = cyclic[:, :3] - cyclic[:, 3:]
    # Separate saturating direct response from the shared cyclic interaction.
    # Penalize departures from cyclic symmetry and secondary operator terms.
    products = quad[:, COLS]
    reg = np.r_[np.zeros(6), np.full(3, EQUAL),
                np.full(products.shape[1], PENALTY), 0., PENALTY]
    def fit(q, predict=False):
        den = 1 + q*d*d
        design = np.c_[base[:, :5]/den[:, None], total, contrasts,
                       products, d/den, d**3/den**EXP]
        scale = np.sqrt(np.mean(design[:nt]**2, axis=0)).clip(1e-8)
        design = design/scale
        a = design[:nt]
        coef = np.linalg.solve(a.T @ a + np.diag(reg + 1e-10), a.T @ y)
        err = a @ coef - y
        return design[nt:] @ coef if predict else err @ err + np.dot(reg, coef*coef)
    q = minimize_scalar(fit, bounds=(.10, .7), method='bounded').x
    return fit(q, True)


'''; new=new.replace('COLS',repr(cols)).replace('EQUAL',repr(equal)).replace('PENALTY',repr(penalty)).replace('EXP',repr(exp)); path.write_text(orig[:orig.index('def _scalar(')]+new+orig[orig.index('def solve_batch('):]); print('promoted',cols,penalty,exp,equal)

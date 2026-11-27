import contextlib, io
import numpy as np
# Reuse the experiment's three archived axis operators and measured observations.
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/walls04.py').read().split('for a in range(3):')[0])
models = {x+y: forward(axes[0][a],axes[1][b]) for a,x in enumerate('RPO') for b,y in enumerate('RPO')}
# The held-out rows never enter wall selection, nuisance projection, or source estimation.
I = np.eye(next(iter(models.values())).shape[1]); IZ = np.eye(Z.shape[1])
def mse(A, tr, te):
    B = Z[tr]; joined = np.column_stack((A[tr],y[tr]))
    K = np.linalg.solve(B.T@B+s['nuisance_ridge']*IZ,B.T@joined)
    projected = joined-B@K; D,t=projected[:,:-1],projected[:,-1]
    w,_=robin._box_simplex(D.T@D+s['source_ridge']*I,-D.T@t,s['u_box'],s['total_budget'])
    e=A[te]@w+Z[te]@(K[:,-1]-K[:,:-1]@w)-y[te]
    return np.mean(e*e)
for name,outer,inner in [('sensor',sensors,shifts),('shift',shifts,sensors%4)]:
    selected=[]; nested=[]; fixed=[]; margin=[]
    for fold in np.unique(outer):
        test=outer==fold; train=~test
        scores=sorted((np.mean([mse(A,train&(inner!=f),train&(inner==f)) for f in np.unique(inner[train])]),law) for law,A in models.items())
        v,law=scores[0]; selected.append(law); margin.append(np.sqrt(scores[1][0])-np.sqrt(v))
        nested.append(np.sqrt(mse(models[law],train,test))); fixed.append(np.sqrt(mse(models['RR'],train,test)))
    print(name,'selected counts',{k:selected.count(k) for k in sorted(set(selected))},'minimum inner RMSE margin',min(margin),'worst outer nested/fixed RMSE',max(nested),max(fixed))

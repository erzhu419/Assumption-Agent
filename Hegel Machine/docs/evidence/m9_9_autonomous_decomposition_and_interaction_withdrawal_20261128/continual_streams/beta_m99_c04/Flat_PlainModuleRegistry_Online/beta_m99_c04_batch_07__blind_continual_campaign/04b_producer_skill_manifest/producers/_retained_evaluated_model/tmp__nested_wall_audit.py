import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
p='/tmp/compare_chamber.py'
exec(compile(open(p).read().split("for x,z in itertools.product('PRO',repeat=2):")[0],p,'exec'))
rows=np.arange(len(y)); groups=(rows%nsen)//4; shifts=rows//nsen
models={}
for x,z in itertools.product('PRO',repeat=2):
    L=kron(eye(ny),axis('x',x),format='csc')+kron(axis('y',z),eye(nx),format='csc')+V
    models[x+z]=np.vstack([splu(L+mu*I).solve(rhs)[ix+nx*iy] for mu in s['mu_shifts']])
for outer in (groups,shifts):
    errors=[]; picks=[]; regrets=[]
    for g in np.unique(outer):
        test=outer==g; train=~test
        inner=[train & (groups==h) for h in np.unique(groups[train])]
        scores={k:np.mean(np.concatenate([(fit(S,train & ~v,sr)[v]-y[v])**2 for v in inner])) for k,S in models.items()}
        best=min(scores,key=scores.get); picks.append(best)
        errs={k:np.sqrt(np.mean((fit(S,train,sr)[test]-y[test])**2)) for k,S in models.items()}
        errors.append(errs[best]); regrets.append(errs[best]-min(errs.values()))
    print('outer', 'rosette' if outer is groups else 'shift', 'selected',picks,'worst outer RMSE',max(errors),'worst excess over outer oracle',max(regrets))

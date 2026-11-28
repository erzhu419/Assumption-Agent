exec(open('/tmp/check_candidate.py').read().split('ys={')[0])
from itertools import combinations
splits += [(f'extrap{i}', np.argsort(z[:,i])[-8:]) for i in (0,1)]
splits += [('leverage',np.argsort(np.sum((S@np.linalg.pinv(S.T@S))*S,axis=1))[-12:])]
# Squared sums/differences cover pure and mixed quadratic directions, including
# directions that cancel at first order in an omitted rational denominator.
directions=np.column_stack([z, *[z[:,a]+sign*z[:,b] for a,b in combinations(range(4),2) for sign in (-1,1)]])
rational=[S]+[S/(1+a*q[:,None]**2) for q in directions.T for a in (.002,.01,.05,.2)]
augmented=[S]+[np.column_stack((S,q**2)) for q in directions.T]+[np.column_stack((S,z[:,a]*z[:,b]*z[:,c])) for a,b,c in combinations(range(4),3)]

def select(models,i):
    # Analytic leave-one-out selects the grammar within the outer training set.
    losses=[]
    for X in models:
        U,d,_=np.linalg.svd(X[i],full_matrices=False)
        U=U[:,d>d[0]*1e-10]
        e=y[i]-U@(U.T@y[i]); h=np.sum(U**2,axis=1)
        losses.append(np.mean(np.abs(e/(1-h))))
    k=np.argmin(losses)
    return k,models[k]

errs={k:[] for k in ('paired','augmented','rational','no_intercept','fixed18_no_intercept','exp2','affine')}
choices={k:[] for k in ('augmented','rational')}
for name,j in splits:
    i=np.setdiff1d(np.arange(n),j)
    for key,models in [('augmented',augmented),('rational',rational)]:
        k,F=select(models,i); choices[key].append(k)
        errs[key].append(np.mean(abs(F[j]@m._ls(F[i],y[i])-y[j])))
    out=m.solve_batch({**s,'train':{'x':x[i],'y':y[i],'traj':T[i]},'test':{'x':x[j]}})
    errs['paired'].append(np.mean(abs(out['y_hat']-y[j])))
    errs['no_intercept'].append(np.mean(abs(S[j,1:]@m._ls(S[i,1:],y[i])-y[j])))
    errs['fixed18_no_intercept'].append(np.mean(abs(z[j]@m._ls(z[i],(y-.18*C)[i])+.18*C[j]-y[j])))
    errs['exp2'].append(np.mean(abs(out['state_pred']-T[j]),axis=0))
    errs['affine'].append(np.mean(abs(D[j]@m._ls(D[i],T[i])-T[j]),axis=0))
for key,v in errs.items():
    a=np.asarray(v); j=np.unravel_index(a.argmax(),a.shape)
    print(key,'mean',round(a.mean(),7),'worst',round(a[j],7),splits[j[0]][0],j[1:],'upper tails/leverage',np.asarray(a[-3:]).reshape(3,-1).mean(1))
    if key in choices: print('  nonbaseline selections',np.count_nonzero(choices[key]),'of',len(splits))
print('Panel B singular values',np.linalg.svd(m._calibration(s)[2][1:],compute_uv=False))
print('Study fields',sorted(s),'training fields',sorted(s['train']),'telemetry shape',np.shape(s['telemetry_vec']))

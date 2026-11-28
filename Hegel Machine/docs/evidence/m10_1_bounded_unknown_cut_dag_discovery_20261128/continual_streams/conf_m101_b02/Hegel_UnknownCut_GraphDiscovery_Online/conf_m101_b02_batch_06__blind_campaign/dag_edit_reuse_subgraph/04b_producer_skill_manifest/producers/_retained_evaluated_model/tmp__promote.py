from pathlib import Path
from refine import X,Z,B,C,D,Q,Y,splits,orig,pred,evaluate,np
from itertools import product

path=Path('/holdout_task/node_01.py')
source=path.read_text()
settings=[(.008,1.),(.004,1.),(.002,1.)]
fold_results=[]
for j in splits:
    i=np.setdiff1d(np.arange(len(Y)),j); order=np.r_[i,j]
    k=orig._kernels(*(a[order] for a in (X,Z,B,C,D)),len(i))
    estimates=[]
    for penalty,_ in settings:
        blocks=[pred(k,Y[i,:2],.004,penalty,'ml')]
        blocks.extend(pred(k,Y[i,c:c+1],noise,penalty,'ml') for c,noise in ((2,.004),(3,.04)))
        estimates.append(np.column_stack(blocks))
    fold_results.append(estimates)

# Prefer broadly accurate settings that clear both public tolerances.
mixtures=[(1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(.5,.5,0.),(.5,0.,.5),(0.,.5,.5),(.3333333333333333,)*3]
ranks=[]
for mixture in mixtures:
    predictions=[sum(w*a for w,a in zip(mixture,estimates) if w) for estimates in fold_results]
    es=np.array([evaluate(a,j)[:2] for a,j in zip(predictions,splits)])
    print('weights',mixture,'public',es[0].round(6),'worst random',es[1:].max(0).round(6),flush=True)
    if max(es[0])<.038:
        ranks.append((es[1:].max(),es[1:].mean(),sum(w>0 for w in mixture),mixture,predictions))
_,_,_,chosen,predictions=min(ranks,key=lambda r:r[:3])
print('promoted',chosen,'worst horizon MAE',max(abs((a[:,1:]-Y[j,1:])@Q.T).mean(0).max() for a,j in zip(predictions,splits)),flush=True)
source=source.replace('def _predict(k, y, noise):', 'def _predict(k, y, noise, penalty):')
source=source.replace('np.full(m,.007)', 'np.full(m,1./m)').replace('v@flat','np.einsum(\'i,ij->j\',v,flat)').replace('flat@(ic-alpha@alpha.T/y.shape[1]).ravel()', 'np.einsum(\'ij,j->i\',flat,(ic-alpha@alpha.T/y.shape[1]).ravel())')
source=source.replace('.035*(delta@delta),grad+.07*delta','penalty*(delta@delta),grad+2*penalty*delta').replace('(-13.,3.)','(-15.,3.)').replace("'maxiter':75,'ftol':1e-7","'maxiter':85,'ftol':2e-7")
a=source.index('    y_hat=_predict'); b=source.index('    final=state_pred',a)
penalties=tuple((p,w) for (p,_),w in zip(settings,chosen) if w)
source=source[:a]+'''    times=s['eval_times']
    h=np.column_stack((-np.expm1(-1.56*times),-np.expm1(-.38*times),np.ones(len(times))))
    q=np.linalg.qr(h)[0]
    modes=train['traj']@q
    # Share covariance relevance between the scalar and dominant time mode.
    joint=np.column_stack((train['y'],modes[:,0]))
    prediction=0.
    for penalty, weight in '''+repr(penalties)+''':
        prediction=prediction+weight*np.column_stack((
            _predict(k,joint,.004,penalty),
            _predict(k,modes[:,1:2],.004,penalty),
            _predict(k,modes[:,2:3],.04,penalty)))
    y_hat=prediction[:,0]
    state_pred=prediction[:,1:]@q.T
'''+source[b:]
path.write_text(source)

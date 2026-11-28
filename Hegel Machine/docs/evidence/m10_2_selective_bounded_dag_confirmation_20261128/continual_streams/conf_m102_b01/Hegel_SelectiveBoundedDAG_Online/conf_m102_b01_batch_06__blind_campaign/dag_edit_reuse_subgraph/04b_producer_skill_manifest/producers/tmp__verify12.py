exec(open('/tmp/fit07.py').read().split('for name,b')[0]);
from pathlib import Path
src=Path('/tmp/v02_node07.py').read_text().split('\ndef run_node(')[0]+Path('/tmp/gp07.txt').read_text()
ns={}; exec(src,ns); ns['_adaptive_orig']=ns['_adaptive']
# Rank challengers on additional folds, before the official verification.
loss=np.zeros((7,4,2))
for j,(a,v) in enumerate(folds[1:]):
    cache={i:ns['_adaptive_orig'](w[np.r_[a,v]],y[a],tr[a],p,i) for i in range(3)}
    ns['_adaptive']=lambda w,y,tr,p,i:cache[i]
    for i in range(7):
        yp,tp=ns['_ensemble'](w[np.r_[a,v]],y[a],tr[a],p,i)
        loss[i,j]=abs(yp-y[v]).mean(),abs(tp-tr[v]).mean()
print('additional CV means by head',loss.mean(1),'worst by head',loss.max(1),flush=True)
rank=np.argsort(.65*loss.mean(1).sum(1)+.35*loss.max(1).sum(1))
# Promote the best independently ranked configuration; retain the established head as a fallback.
i=int(rank[0]); print('chosen configuration',i,flush=True)
# Remove the superseded broad kernel implementation from the deployed node.
a=src.index('\ndef _fit('); b=src.index('\nfrom scipy.optimize import minimize\n'); src=src[:a]+src[b:]
src=src.replace('from scipy.optimize import minimize_scalar\n','')
src+='\n\ndef run_node(inputs: dict) -> dict:\n    study = inputs["__input__"]\n    train = study["train"]\n    w = np.asarray(inputs["node_00"]["w"])\n    y, tr = _ensemble(w, np.asarray(train["y"]), np.asarray(train["traj"]), study["panel_c"], '+str(i)+')\n    tail = tr[:, -1]\n    return {"y_hat": y, "state_pred": tr, "diagnostics_vec": np.array([y.mean(), y.std(), tail.mean(), tail.std()])}\n'
Path('/holdout_task/node_01.py').write_text(src)

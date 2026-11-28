exec(open('/tmp/stress_batch11.py').read().split('worst=[]')[0])
from scipy.optimize import least_squares
from itertools import combinations
np.set_printoptions(precision=6)
print('study keys:',sorted(s),'train keys:',sorted(tr),'telemetry shape:',np.shape(s['telemetry_vec']))
F=a['_readout'](w); Y=tr['y']; T=tr['traj']; pair=F[:,-1]; X=w
for sign in (-1,1):
    splits.append((f'pair-tail{sign:+d}',np.argsort(sign*pair)[:-12],np.argsort(sign*pair)[-12:]))
D=np.column_stack([X[:,k]*X[:,l] for k,l in combinations(range(4),2)])
rivals={'tanh':(F,lambda q:q),'rational-index':(F,lambda q:q),'rational-pair':(F,lambda q:pair),'all-cross':(np.column_stack((aff(X),D)),lambda q:q),'pair-difference':(np.column_stack((aff(X),X[:,0]*X[:,2]-X[:,1]*X[:,3])),lambda q:q),'no-panel-B-pair':(a['_readout'](z),lambda q:q)}
for name,(features,den) in rivals.items():
    errs=[]
    for label,i,j in splits:
        if name.startswith('rational'):
            def pred(p):
                q=features@p[:-2]; return q+p[-2]*q/(1+(np.exp(p[-1])*den(q))**2)
            init=np.r_[.8*a['_fit'](features[i],Y[i]),.5,0.]
            m=least_squares(lambda p:(pred(p)-Y)[i],init,max_nfev=220,bounds=(np.r_[np.full(features.shape[1],-np.inf),0.,-3.],np.r_[np.full(features.shape[1],np.inf),np.inf,3.]))
            yh=pred(m.x)[j]
        else: yh=a['_scalar_readout'](features[i],Y[i],features[j])
        errs.append(np.mean(abs(yh-Y[j])))
    print(name,'public / avg / worst / worst name:',errs[0],np.mean(errs),max(errs),splits[np.argmax(errs)][0])
for rank in (1,2,3,5):
    stats=[]
    for _,i,j in splits:
        A=aff(X[i]); coef=a['_fit'](A,T[i]); modes=np.linalg.svd(A@coef,full_matrices=False)[2][:rank].T
        err=abs(aff(X[j])@coef@modes@modes.T-T[j]); stats.append([err.mean(),err.mean(0).max()])
    print('trajectory rank',rank,'public / worst slice / worst horizon:',stats[0][0],*np.max(stats,axis=0))
pa=s['panel_a']; ref=np.arange(len(pa['u'])); i,j=splits[0][1:]; train={k:(v[i] if k!='summary' else v) for k,v in tr.items()}; metrics=[]
for omit in np.array_split(ref,6):
    keep=np.setdiff1d(ref,omit); out=solve(dict(s,panel_a={k:v[keep] for k,v in pa.items()},train=train,test={'x':x[j]})); yh,th=check(out,len(j)); metrics.append([np.mean(abs(yh-Y[j])),np.mean(abs(th-T[j]))])
print('reference block omission: worst y / trajectory MAEs',np.max(metrics,axis=0)); assert np.max(metrics,axis=0)[0] <= .038

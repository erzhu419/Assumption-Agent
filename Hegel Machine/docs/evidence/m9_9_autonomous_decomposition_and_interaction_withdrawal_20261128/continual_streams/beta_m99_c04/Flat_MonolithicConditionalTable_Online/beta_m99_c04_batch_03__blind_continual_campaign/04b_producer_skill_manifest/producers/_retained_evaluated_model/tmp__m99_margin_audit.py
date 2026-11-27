import importlib.util, pickle
import numpy as np
p='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('candidate',p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
items=pickle.load(open('/holdout_task/public_batch.pkl','rb'))

def fit_rational(x,y,base,qs):
    # Train-only selection: allow saturation along either signed neighbor contrast.
    best=(float('inf'),None,None,None)
    for qid,q in enumerate(qs):
        for a in (0.,.001,.01,.05,.15,.5):
            D=base/(1+a*q[:,None]**2)
            b=np.linalg.lstsq(D,y,rcond=None)[0]
            loss=np.linalg.norm(D@b-y)
            if loss<best[0]: best=(loss,qid,a,b)
    return best[1:]

def contrasts(x,j):
    t=x[:,j]; prev=x[:,(j-1)%4]; nxt=x[:,(j+1)%4]
    return [t]+[t+c*v for v in (prev,nxt) for c in (-1.,-.5,.5,1.)]

for item in items:
    s=item['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; rows=np.arange(len(x)); pure=np.count_nonzero(x,axis=1)==1
    # Hold out whole single-channel sweep families, then split the mixed sweeps.
    folds=[rows[pure & (x[:,j]!=0)] for j in range(4)]
    folds+=list(np.array_split(np.random.default_rng(481).permutation(rows[~pure]),5))
    stats={k:np.empty_like(y) for k in ('cyclic','expanded','rational_no_cubic','rational_cyclic')}
    for test in filter(len,folds):
        tr=np.setdiff1d(rows,test); b=m._fit_rosette(x[tr],y[tr]); train=np.column_stack([m._rosette_basis(x[tr],j)@b[j] for j in range(4)])
        e=np.linalg.norm(train-y[tr],axis=1); keep=tr[e<=5*max(np.median(e),1e-12)]
        for j in range(4):
            base=m._rosette_basis(x,j); stats['cyclic'][test,j]=base[test]@b[j]-y[test,j]
            t=x[:,j]; prev=x[:,(j-1)%4]; nxt=x[:,(j+1)%4]; opp=x[:,(j+2)%4]
            expanded=np.column_stack([base,opp,t*prev,prev*nxt,t*opp,t*prev*nxt,(t-prev)**2,(t+nxt)**2])
            stats['expanded'][test,j]=expanded[test]@np.linalg.lstsq(expanded[keep],y[keep,j],rcond=None)[0]-y[test,j]
            qs=contrasts(x,j)
            for name,D in [('rational_no_cubic',base[:,:5]),('rational_cyclic',base)]:
                idx,a,v=fit_rational(x[keep],y[keep,j],D[keep],[q[keep] for q in qs]); stats[name][test,j]=(D[test]@v)/(1+a*qs[idx][test]**2)-y[test,j]
    gross=np.linalg.norm(stats['cyclic'],axis=1)>3
    print('gross rows',rows[gross].tolist())
    for name,e in stats.items():
        print(name,'max clean single/mixed',*[float(np.max(abs(e[mask]))) for mask in (pure&~gross,~pure&~gross)],'clean RMSE',float(np.sqrt(np.mean(e[~gross]**2))))
    S,U=m._design(s); y=m._calibrate(s); z=s['Z_nuisance']; cap=s['u_box']; budget=s['total_budget']; r=s['source_ridge']; nr=s['nuisance_ridge']; zdim=z.shape[1]
    by_sensor=np.tile(np.arange(len(s['sensor_ij'])),len(s['mu_shifts']))
    scores=[]
    for sensor in np.unique(by_sensor):
        te=by_sensor==sensor; tr=~te; zz=z[tr]; pz=np.linalg.solve(zz.T@zz+nr*np.eye(zdim),zz.T)
        pair=[]
        for profile in (False,True):
            X=S[tr]; yy=y[tr]
            if profile: X=X-zz@(pz@X); yy=yy-zz@(pz@yy)
            w,nu=m._box_simplex(X.T@X+r*np.eye(X.shape[1]),X.T@yy,cap,budget)
            yhat=S[te]@w
            if profile: yhat+=z[te]@(pz@(y[tr]-S[tr]@w))
            pair.append(np.sqrt(np.mean((yhat-y[te])**2)))
        scores.append(pair)
    print('leave-physical-sensor-out RMSE worst unprofiled/profiled',np.max(scores,axis=0),'median',np.median(scores,axis=0))

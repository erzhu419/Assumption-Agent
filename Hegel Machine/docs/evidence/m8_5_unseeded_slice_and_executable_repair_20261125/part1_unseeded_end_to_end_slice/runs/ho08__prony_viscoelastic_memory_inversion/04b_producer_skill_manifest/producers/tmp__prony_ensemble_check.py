import pickle
import numpy as np
import prony_ensemble as m
from prony_stresscheck import strain, probe, clean


def compare(e, truth, G):
    t,x,y,u,z=[e[k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain')]
    tr=m._History(t,x); te=m._History(u,z)
    D=np.c_[np.ones(len(t)),(t-t.mean())/np.ptp(t)]
    grid=np.linspace(*m._LOG_BOUNDS,61); A=tr.columns(grid)[0]; keep,g=m._screen(A,D,y)
    rows, cols=m._spectrum(tr,grid,A,D,y,keep,g); score=np.array([a[0] for a in rows]); pred=[]; gs=[]
    for i,(_,v,res) in enumerate(rows):
        # _spectrum's closure returns projected columns for any log-time vector.
        pred.append(te.columns(v[:i])[0]@v[i:]); gs.append(m._equilibrium(v,res,cols,keep.sum()))
    pred=np.array(pred); gs=np.array(gs)
    inds=score.argmin(); accum=[]
    for temp in (0.,1.,2.,4.):
        w=(np.arange(len(score))==inds).astype(float) if not temp else np.exp(-(score-score.min())/(2*temp)); w/=w.sum()
        accum.append((np.linalg.norm(w@pred-truth)/np.linalg.norm(truth),abs(w@gs/G-1)))
    return np.array(accum)

if __name__=='__main__':
    with open('/holdout_task/public_rheology.pkl','rb') as f: p=pickle.load(f)
    print('columns are hard BIC, BIC weights, temperature 2, temperature 4',flush=True)
    print('public [stress, G]', compare(p['experiment'],p['gt_test_stress'],p['gt_G_inf']).T.round(6),flush=True)
    for name,tau,g,noise in [('single',[.07],[11.,22.],.015),('short',[.1,.85,4.],[9.,21.,14.,8.],.015),('moderate',[.26,1.7,9.],[10.,21.,14.,8.],.015),('slow',[.63,3.2,16.],[10.,21.,14.,8.],.015),('four',[.06,.36,2.1,12.],[14.,15.,13.,11.,8.],.005),('wide',[.12,.5,13.],[7.,35.,10.,12.],.008)]:
        rows=[]
        for j in range(8):
            rng=np.random.default_rng(1190+j); t=np.linspace(0,30,601); u=np.linspace(0,24,481)
            if j%2:
                t[1:-1]+=rng.uniform(-.012,.012,599); u[1:-1]+=rng.uniform(-.018,.018,479)
            x=strain(t); z=probe(u); gg=np.array(g); y=clean(t,x,tau,gg); gt=clean(u,z,tau,gg)
            y+=rng.uniform(-1,1)+rng.uniform(-.25,.25)*t+rng.normal(0,noise,len(t))
            ix=rng.choice(len(t),12,replace=False); y[ix]+=rng.choice([-1,1],12)*rng.uniform(.3,2.,12)
            rows.append(compare(dict(train_t=t,train_strain=x,train_stress_raw=y,test_t=u,test_strain=z),gt,g[0]))
        r=np.array(rows); print(name,'worst',r.max(0).T.round(4).tolist(),'median',np.median(r,axis=0).T.round(4).tolist(),flush=True)

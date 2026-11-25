import sys, time
sys.path.insert(0, '/holdout_task')
import numpy as np
import prony_rheology as model

# Direct interval integral, independent of the production state recurrence.
def clean(t, x, tau, g):
    dt = np.diff(t)
    delta = np.diff(x)
    tau = np.asarray(tau)
    end_state = (delta / dt)[:, None] * tau * -np.expm1(-dt[:, None] / tau)
    y = g[0]*x.copy()
    for i in range(1, len(t)):
        y[i] += np.sum((np.exp(-(t[i]-t[1:i+1, None])/tau)*end_state[:i]) @ g[1:])
    return y

def strain(t):
    return .05*np.minimum(t/2, 1) + np.where(t >= 14, .03*np.sin(2*np.pi*.095*(t-14)), 0.)

def probe(t):
    return .025*np.sin(2*np.pi*(.13*t+.022*t*t)) + .025*(t>=9.) - .015*(t>=18.)

if __name__ == '__main__':
    for name, tau, g, noise in [
        ('single', [.07], [11.,22.], .015),
        ('short', [.10,.85,4.], [9.,21.,14.,8.], .015),
        ('moderate', [.26,1.7,9.], [10.,21.,14.,8.], .015),
        ('slow', [.63,3.2,16.], [10.,21.,14.,8.], .015),
        ('slow-clean', [.63,3.2,16.], [10.,21.,14.,8.], .0001),
        ('four', [.06,.36,2.1,12.], [14.,15.,13.,11.,8.], .005),
    ]:
        rows=[]
        for j in range(8):
            start=time.perf_counter(); rng=np.random.default_rng(1190+j)
            t=np.linspace(0,30,601); u=np.linspace(0,24,481)
            if j%2:
                t[1:-1]+=rng.uniform(-.012,.012,len(t)-2); u[1:-1]+=rng.uniform(-.018,.018,len(u)-2)
            x=strain(t); z=probe(u); gt=clean(u,z,tau,np.asarray(g)); y=clean(t,x,tau,np.asarray(g))
            y+=rng.uniform(-1,1)+rng.uniform(-.25,.25)*t+rng.normal(0,noise,len(t))
            ix=rng.choice(len(t),12,replace=False); y[ix]+=rng.choice([-1,1],12)*rng.uniform(.3,2.,12)
            e={'train_t':t,'train_strain':x,'train_stress_raw':y,'test_t':u,'test_strain':z}
            out=model.fit_prony_and_predict_stress(e)
            train=model._History(t,x); d=np.c_[np.ones(len(t)),(t-t.mean())/np.ptp(t)]; grid=np.linspace(*model._LOG_BOUNDS,61); a=train.columns(grid)[0]; keep,gg=model._screen(a,d,y); v,_,_=model._spectrum(train,grid,a,d,y,keep,gg); m=(len(v)-1)//2
            err=np.linalg.norm(out['test_stress']-gt)/np.linalg.norm(gt)
            rows.append([err,abs(out['G_inf']/g[0]-1),abs(v[m]/g[0]-1),m,time.perf_counter()-start,len(set(ix)&set(np.flatnonzero(~keep)))])
        r=np.array(rows)
        for j,sub in [('uniform',r[::2]),('jittered',r[1::2])]:
            print(name,j,'median/worst [stress,G,Graw]',np.round(np.median(sub[:,:3],0),5),np.round(np.max(sub[:,:3],0),5),'modes',sub[:,3].astype(int),'max_seconds_two_fits',round(sub[:,4].max(),3),'spike_recall_min',int(sub[:,5].min()),flush=True)

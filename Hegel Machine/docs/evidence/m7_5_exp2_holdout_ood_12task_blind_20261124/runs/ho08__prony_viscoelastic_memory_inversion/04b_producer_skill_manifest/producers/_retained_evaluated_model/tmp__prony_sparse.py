import pickle
import warnings
import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.signal import lfilter
from scipy.special import softmax
import prony_rheology as p


def design(t, s, tau):
    dt = np.diff(t)
    if np.max(abs(dt-dt[0])) > 1e-9*dt[0]:
        return p._design(t, s, tau)
    h = dt[0]/tau
    ds = np.diff(s, prepend=s[0])
    return np.column_stack([s, *(lfilter([-np.expm1(-z)/z], [1., -np.exp(-z)], ds) for z in h)])


def candidates(t, s, y, training=None):
    if training is None:
        training = np.ones(len(t), bool)
    d = np.c_[np.ones_like(t), (t-t.mean())/np.ptp(t)]
    grid = np.geomspace(.02,20,60)
    a = design(t,s,grid)
    g0,r = p._calibrate(a[training],d[training],y[training])
    sig = p._noise(r)
    keep = training.copy()
    keep[training] = abs(r-np.median(r)) <= 4.5*sig
    q = np.linalg.qr(d[keep], mode='reduced')[0]
    b = y[keep] - q@(q.T@y[keep])
    ap = a[keep]-q@(q.T@a[keep])
    magnitude = max(np.linalg.norm(ap@g0)/np.linalg.norm(s[keep]-np.mean(s[keep])), np.finfo(float).eps)
    result = {'grid': [(1, grid,g0)]}

    peaks = []
    for i in np.flatnonzero(g0[1:] > 1e-5*np.sum(g0)):
        m = g0[i+1]*np.linalg.norm(ap[:,i+1]); z = np.log(grid[i])
        if peaks and z-peaks[-1][0]/peaks[-1][1] < .42:
            peaks[-1] = (peaks[-1][0]+m*z,peaks[-1][1]+m)
        else:
            peaks.append((m*z,m))
    ranked = [z/m for z,m in sorted(peaks,key=lambda zm:-zm[1])]

    def model(z, strength, full=False):
        tau = np.exp(z)
        c = design(t,s,tau)[keep]
        c -= q@(q.T@c)
        row = (strength*sig/magnitude)*np.r_[0.,tau/np.ptp(t)]
        A = np.vstack((c,row))
        norms = np.linalg.norm(A,axis=0)
        h = nnls(A/norms, np.r_[b,0.],maxiter=100)[0]/norms
        residual = b-c@h
        rr = np.r_[residual,row@h]
        return (tau,h,rr,residual) if full else rr

    for strength in (0, 6, 12):
        fits=[]
        for n in (1,2,3,4):
            defaults = {1:[1.],2:[.2,5.],3:[.13,1.2,8.],4:[.035,.22,1.6,10.]}
            starts = [np.log(defaults[n])]
            if len(ranked)>=n and np.max(abs(np.sort(ranked[:n])-starts[0])) > .15:
                starts.append(np.sort(ranked[:n]))
            if strength and len(starts)<3:
                starts.append(np.log(result[f'm{n}_0'][0][1]))
            best = None
            for start in starts:
                try:
                    opt = least_squares(model,np.clip(start,np.log(.02),np.log(20.)),args=(strength,),bounds=(np.log(.02),np.log(20.)),max_nfev=45,ftol=2e-7)
                    fitted = model(opt.x,strength,True)
                except (RuntimeError, np.linalg.LinAlgError):
                    continue
                if best is None or fitted[2]@fitted[2] < best[2]@best[2]:
                    best = fitted
            if best is None:
                best = (grid,g0,np.r_[b-ap@g0,0],b-ap@g0)
            tau,g,rr,r = best
            fits.append((tau,g,rr@rr,r@r))
            result[f'm{n}_{strength}'] = [(1,tau,g)]
        for k in (2., 4., np.log(len(b)), 10.):
            score = np.array([loss/sig**2+k*n for n,(_,_,loss,_) in enumerate(fits,1)])
            result[f'avg{k:.1f}_{strength}'] = [(w,tau,g) for w,(tau,g,_,_) in zip(softmax(-.5*score),fits) if w>1e-6]
    return result,keep


def main():
    e0=pickle.load(open('/holdout_task/public_rheology.pkl','rb'))
    t,s,y,u,v=(e0['experiment'][k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain'))
    step=.035*(np.clip((u-1)/.15,0,1)-1.4*np.clip((u-9)/.1,0,1)+.6*np.clip((u-17)/.1,0,1))
    e = [ ('public', t,s,y, e0['gt_G_inf'], [v], [e0['gt_test_stress']]) ]
    rng=np.random.default_rng(285409)
    for slow in (3.,6.,9.,13.,18.):
        for j in range(10):
            dt=[.04,.05,.075,.10,.05][j%5]; noise=[.008,.016,.025,.035,.015][j%5]
            tt=np.arange(0,30+dt/2,dt)
            sw=rng.uniform(12,16)
            st=np.where(tt<=sw,.05*np.minimum(tt/rng.uniform(1.4,3.4),1),.05+.03*np.sin(rng.uniform(.2,.55)*(tt-sw)))
            tau=np.sort([rng.uniform(.07,.5),rng.uniform(.75,2.3),slow]); g=np.r_[rng.uniform(8,24),rng.uniform([16,8,4],[36,24,16])]
            if j%3==0:
                tau=np.r_[rng.uniform(.022,.055),tau]; g=np.r_[g[0],rng.uniform(5,18),g[1:]]
            if j==8:
                tau=tau[1:];g=np.delete(g,1)
            yy=design(tt,st,tau)@g+rng.uniform(-3,3)+rng.uniform(-.2,.2)*tt+rng.normal(0,noise,len(tt))
            idx=rng.choice(len(tt),round(.02*len(tt)),False); yy[idx]+=rng.uniform(.4,5,len(idx))*rng.choice([-1,1],len(idx))
            tests=[design(u,w,tau)@g for w in (v,step)]
            e.append((slow,tt,st,yy,g[0],[v,step],tests))
    result={}
    for label,t,s,y,g0,strains,truths in e:
        cs,keep=candidates(t,s,y)
        for name,models in cs.items():
            gg=sum(w*g[0] for w,_,g in models)
            errors=[np.linalg.norm(sum(w*(design(u,sv,tau)@g) for w,tau,g in models)-yt)/np.linalg.norm(yt) for sv,yt in zip(strains,truths)]
            result.setdefault(name,[]).append([label,abs(gg-g0)/g0,errors[0],errors[-1],gg])
    for name,rows in result.items():
        if name.startswith(('m1','m2')):continue
        val=np.array([r[1:4] for r in rows[1:]])
        print(name,'public',np.round(100*np.array(rows[0][1:3]),2),'synthetic failures G/chirp/step',np.sum(val>[.06,.022,.022],axis=0),'p90',np.round(100*np.quantile(val,.9,axis=0),2),'worst',np.round(100*np.max(val,axis=0),2),'G medians by slow bin',[round(100*np.median(val[i:i+10,0]),2) for i in range(0,50,10)])

if __name__ == '__main__':
    main()

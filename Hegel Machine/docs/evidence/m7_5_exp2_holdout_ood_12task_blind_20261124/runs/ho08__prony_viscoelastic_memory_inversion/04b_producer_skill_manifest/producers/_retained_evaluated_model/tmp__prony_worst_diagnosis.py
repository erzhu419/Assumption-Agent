import pickle
from collections import defaultdict
import numpy as np
from scipy.optimize import nnls
from prony_sparse import design
import prony_rheology as p

TAU = np.geomspace(.02, 20, 60)
SPEC = [('grid',0,0,0), ('slow6',6,0,0), ('slow12',12,0,0), ('curved12',0,12,0), ('curved16',0,16,0), ('curved20',0,20,0), ('both',5,10,0)]
SPEC += [(f'{name}_mass{mass}',a,b,mass) for name,a,b,_ in SPEC if name in ('grid','slow6','curved16','both') for mass in (0.5,1.5,3.)]

def estimate(t,s,y):
    d=np.c_[np.ones_like(t),(t-t.mean())/np.ptp(t)]
    a=design(t,s,TAU)
    g,r=p._calibrate(a,d,y)
    sig=p._noise(r); keep=abs(r-np.median(r))<=4.5*sig
    q=np.linalg.qr(d[keep],mode='reduced')[0]
    b=y[keep]-q@(q.T@y[keep]); ap=a[keep]-q@(q.T@a[keep])
    mag=max(np.linalg.norm(ap@g)/np.linalg.norm(s[keep]-s[keep].mean()), np.finfo(float).eps)
    x=TAU/np.ptp(t)
    coeff={}
    for name,slow,curved,mass in SPEC:
        if name=='grid':
            coeff[name]=g; continue
        penalty=np.array([np.r_[0,slow*x],np.r_[0,curved*x*x],np.r_[0,np.full(len(x),mass)]])*sig/mag
        aa=np.vstack([ap,penalty]); scale=np.maximum(np.linalg.norm(aa,axis=0),1e-14)
        coeff[name]=nnls(aa/scale,np.r_[b,0.,0.,0.],maxiter=1200)[0]/scale
    return coeff,keep

def run():
    x=pickle.load(open('public_rheology.pkl','rb')); e=x['experiment']; t,s,y,u,v=(e[k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain'))
    step=.035*(np.clip((u-1)/.15,0,1)-1.4*np.clip((u-9)/.1,0,1)+.6*np.clip((u-17)/.1,0,1)); aa=[design(u,w,TAU) for w in (v,step)]
    out=defaultdict(list); pub={}; oracle=[]; diagnostics=[]
    for name,g in estimate(t,s,y)[0].items():
        pub[name]=[abs(g[0]/x['gt_G_inf']-1),np.linalg.norm(aa[0]@g-x['gt_test_stress'])/np.linalg.norm(x['gt_test_stress'])]
    for seed in (285409,78483):
        rng=np.random.default_rng(seed)
        for slow in (3.,6.,9.,13.,18.):
            for j in range(12):
                dt=[.04,.05,.075,.10,.05,.05][j%6]; noise=[.008,.016,.025,.035,.015,.012][j%6]
                tt=np.arange(0,30+dt/2,dt); sw=rng.uniform(12,16)
                st=np.where(tt<=sw,.05*np.minimum(tt/rng.uniform(1.4,3.4),1),.05+.03*np.sin(rng.uniform(.2,.55)*(tt-sw)))
                tau=np.sort([rng.uniform(.07,.5),rng.uniform(.75,2.3),slow]); g=np.r_[rng.uniform(8,24),rng.uniform([16,8,4],[36,24,16])]
                if j%3==0: tau=np.r_[rng.uniform(.022,.055),tau]; g=np.r_[g[0],rng.uniform(5,18),g[1:]]
                if j==8: tau=tau[1:]; g=np.delete(g,1)
                ad=design(tt,st,tau); dd=np.c_[np.ones(len(tt)),tt]; yy=ad@g+rng.uniform(-3,3)+rng.uniform(-.2,.2)*tt+rng.normal(0,noise,len(tt))
                idx=rng.choice(len(tt),round(.02*len(tt)),False); yy[idx]+=rng.uniform(.4,5,len(idx))*rng.choice([-1,1],len(idx))
                target=[design(u,w,tau) for w in (v,step)]; ys=[a@g for a in target]
                vals,keep=estimate(tt,st,yy)
                ts, final = p._estimate(tt,st,yy)
                for name,gg in vals.items():
                    errs=[np.linalg.norm(a@gg-ref)/np.linalg.norm(ref) for a,ref in zip(aa,ys)]
                    out[name].append((slow,j,abs(gg[0]/g[0]-1),*errs))
                clean=np.ones(len(tt),bool);clean[idx]=False
                go, _=p._projected_fit(ad,dd,yy,clean.astype(float))
                errs=[np.linalg.norm(a@(go-g))/np.linalg.norm(ref) for a,ref in zip(target,ys)]
                oracle.append((slow,j,abs(go[0]/g[0]-1),*errs,int(np.sum(keep!=clean))))
                pred = [design(u,w,ts)@final for w in (v,step)]
                errors = [np.linalg.norm(f-ref)/np.linalg.norm(ref) for f,ref in zip(pred,ys)]
                peaks = [(round(a,2),round(b,2)) for a,b in zip(ts,final[1:]) if b>0.05]
                diagnostics.append((abs(final[0]/g[0]-1), max(errors), (seed, j, dt, noise), np.round(tau,3).tolist(), np.round(g,2).tolist(), round(final[0],2), round(go[0],2), peaks, np.round(100*np.array(errors),2).tolist(), np.round(100*np.array(errs),2).tolist()))
    for k,label in enumerate(('modulus','forecast')):
        print('Worst by',label,': errors, seed/index/dt/noise, true tau, true G, v02 G_inf, oracle G_inf, v02 peaks, v02 stress %, oracle stress %')
        for row in sorted(diagnostics, key=lambda row: -row[k])[:6]:
            print(tuple(round(z,4) for z in row[:2]), *row[2:])
if __name__=='__main__': run()

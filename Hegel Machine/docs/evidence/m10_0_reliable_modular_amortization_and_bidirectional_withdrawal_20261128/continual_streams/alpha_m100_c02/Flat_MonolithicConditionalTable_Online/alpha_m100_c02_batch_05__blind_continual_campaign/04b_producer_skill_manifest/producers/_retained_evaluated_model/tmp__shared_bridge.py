import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares


def clean_bridge(c):
    l,r=c['true_channel_loads'],c['bridge_raw']; s=c['single_channel_sweeps']
    good=np.max(abs(r),axis=1)<4
    j=s['lab_id']; good_s=abs(s['cmd_stress'])<4
    return l[good],r[good],j[good_s],s['true_stress'][good_s],s['cmd_stress'][good_s]


def fit_bridge(l,r,j,x,y,n=5,style='tanh',interaction=False):
    u=np.r_[l.ravel(),x]; v=np.r_[np.roll(l,2,axis=1).ravel(),np.zeros_like(x)]; k=np.r_[np.tile(np.arange(4),len(l)),j]; obs=np.r_[r.ravel(),y]
    def selfbasis(u): return chebvander(u/1.7,2*n-1)[...,1::2]
    def crossbasis(v,b):
        if style=='tanh': return np.stack((v,v*np.tanh(b*v)),axis=-1)
        if style=='rat': return np.stack((v,v*v/np.sqrt(1+(b*v)**2)),axis=-1)
        if style=='poly':
            z=v/1.7; a=chebvander(z,8)[...,1:]; return a-chebvander(np.array([0.]),8)[0,1:]
        return np.stack((v,np.logaddexp(b*v,-b*v)/b-np.log(2)/b),axis=-1)
    m=8 if style=='poly' else 2
    def apply(p,u,v,k):
        f=selfbasis(u)@p[7:7+n]+crossbasis(v,p[-1])@p[7+n:7+n+m]
        if interaction: f=f+p[-3]*u*v+p[-2]*u*u*v
        return p[k]+np.r_[1.,p[4:7]][k]*f
    start=np.r_[[-.12,.14,-.11,.12],1.6,.986,1.57,np.zeros(n+m),np.zeros(2*interaction),2.]
    start[7]=1.2
    fit=least_squares(lambda p:apply(p,u,v,k)-obs,start,bounds=(np.r_[np.full(len(start)-1,-np.inf),.05],np.r_[np.full(len(start)-1,np.inf),20.]),ftol=1e-11,gtol=1e-11,xtol=1e-11,max_nfev=300)
    return lambda u,v,k=np.arange(4):apply(fit.x,u,v,k),fit.x


def invert(raw,forward):
    raw=np.asarray(raw); x=np.zeros_like(raw); eps=1e-5
    for i in range(35):
        v=np.roll(x,2,axis=-1); f=forward(x,v)-raw
        a=(forward(x+eps,v)-forward(x-eps,v))/(2*eps); b=(forward(x,v+eps)-forward(x,v-eps))/(2*eps)
        dx=(np.roll(a,2,axis=-1)*f-b*np.roll(f,2,axis=-1))/(a*np.roll(a,2,axis=-1)-b*np.roll(b,2,axis=-1))
        x=np.clip(x-np.clip(dx,-.35,.35),-1.7,1.7)
        if np.max(abs(dx))<1e-10: break
    return x

if __name__=='__main__':
    import pickle
    d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; l,r,j,x,y=clean_bridge(d['calibration_load_cells'])
    for n,style,i in [(n,s,i) for n in [4,5] for s in ['tanh','rat','log','poly'] for i in [False,True]]:
        e=[]; raw=[]
        for z in range(5):
            tr=np.arange(len(l))%5!=z
            f,p=fit_bridge(l[tr],r[tr],j,x,y,n,style,i); e.append((invert(r[~tr],f)-l[~tr]).sum(axis=1)); raw.append(f(l[~tr],np.roll(l[~tr],2,axis=1))-r[~tr]);
        f,p=fit_bridge(l,r,j,x,y,n,style,i)
        print(n,style,i,'worst fold total RMSE',round(max(np.sqrt(np.mean(a*a)) for a in e),6),'raw RMSE',round(np.sqrt(np.mean(np.vstack(raw)**2)),6),'p',p.round(5))
        if style=='poly' and n==5 and not i:
            for k in ['train','test']:
                a=d[k]['past_channel_raw']; b=invert(a,f); print(k,'past range',b.min(),b.max(),'max raw roundtrip',np.max(abs(f(b,np.roll(b,2,axis=-1))-a)))

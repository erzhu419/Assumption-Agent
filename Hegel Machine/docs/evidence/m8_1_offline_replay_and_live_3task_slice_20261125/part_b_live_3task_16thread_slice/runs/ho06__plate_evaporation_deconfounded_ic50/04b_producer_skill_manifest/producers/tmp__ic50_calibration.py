import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

with open('/holdout_task/public_screen.pkl', 'rb') as f:
    screen = pickle.load(f)

def extract(plate):
    wells = plate['wells']
    xy = np.array([(w['row'], w['col']) for w in wells])
    y = np.array([w['fluorescence'] for w in wells])
    kinds = np.array([w['well_type'] for w in wells])
    return wells, xy, y, kinds

def field(xy, p, interaction):
    r,c = xy.T
    d = np.minimum.reduce([r,15-r,c,23-c])
    e = np.exp(-d/np.exp(p[4]))
    illum = p[2]*(r/15-.5)+p[3]*(c/23-.5)
    return p[0]+p[1]*e+illum if not interaction else (p[0]+p[1]*e)*(1+illum)

def fit(xy, z, model, start=None):
    independent, interaction = model
    p0 = np.array([1.,.4,0.,0.,np.log(2.)]) if start is None else start
    if independent:
        return least_squares(lambda p: field(xy,p,interaction)-z,p0,bounds=([0.,0.,-1.,-1.,np.log(.25)],[3.,2.,1.,1.,np.log(12.)]),loss='soft_l1',f_scale=.008).x
    return np.r_[least_squares(lambda p: field(xy,np.r_[p,np.log(2.)],interaction)-z,p0[:4],loss='soft_l1',f_scale=.008).x,np.log(2.)]

for model in [(False,False),(False,True),(True,False),(True,True)]:
    records={}; control_errors=[[],[]]; profiles=[]
    for plate in screen['plates']:
        wells,xy,y,k=extract(plate)
        b=np.median(y[k=='POS_KILL']); s=np.median(y[k=='DMSO'])-b
        dm=k=='DMSO'; x=xy[dm]; z=(y[dm]-b)/s
        p=fit(x,z,model); profiles.append(np.round(np.r_[p[:4],np.exp(p[4])],4).tolist())
        for i in range(len(z)):
            sel=np.arange(len(z))!=i
            q=fit(x[sel],z[sel],model)
            pred=field(x[i:i+1],q,model[1])[0]
            r,c=x[i]; idx=int(min(r,15-r,c,23-c)>0)
            control_errors[idx].append((pred-z[i])/z[i])
        t=k=='COMPOUND'; v=(y[t]-b)/s/field(xy[t],p,model[1]); pos=np.flatnonzero(t)
        for j,vj in zip(pos,v):
            w=wells[j]; records.setdefault(w['compound_id'],[]).append((w['log10_conc_uM'],vj))
    print('model free-decay, interaction',model,'control RMS edge/interior %',np.round([100*np.mean(np.square(v))**.5 for v in control_errors],3),'profiles',profiles)
    for loss in ('soft_l1','huber','cauchy','refit'):
        errors=[]; out=[]; scatter=[]
        for name,obs in records.items():
            c,v=np.array(obs).T
            res=lambda p: expit(-np.log(10)*np.exp(p[1])*(c-p[0]))-v
            bds=([c.min()-5,-3],[c.max()+5,3]); init=[c[np.argmin(abs(v-.5))],0.]
            p=least_squares(res,init,bounds=bds,loss='soft_l1' if loss=='refit' else loss,f_scale=.02).x
            if loss=='refit':
                inliers=abs(res(p))<.05
                p=least_squares(lambda q: res(q)[inliers],p,bounds=bds,loss='soft_l1',f_scale=.015).x
            g=screen['gt'][name]; errors.append((abs(p[0]-g['log10_ic50_uM']),abs(np.exp(p[1])-g['hill_slope'])))
            residual=res(p); out.extend(v[abs(residual)>.07].tolist()); scatter.extend(residual[abs(residual)<=.07].tolist())
        print(loss,'worst errors',np.round(np.max(errors,axis=0),5),'inlier MAD',round(np.median(np.abs(scatter)),4),'outlier count',len(out),'outlier V range',np.round([min(out),max(out)],3) if out else [])

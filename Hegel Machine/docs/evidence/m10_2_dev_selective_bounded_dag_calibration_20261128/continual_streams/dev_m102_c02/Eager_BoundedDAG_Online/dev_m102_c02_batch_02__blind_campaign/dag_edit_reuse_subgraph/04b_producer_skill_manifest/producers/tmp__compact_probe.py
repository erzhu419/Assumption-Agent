from readout_probe import A,c,y,old,folds
from scipy.optimize import least_squares
import numpy as np
from scipy.special import erf

def evaluate(kind,mode,tr,te):
    b=np.linalg.lstsq(np.c_[A,old][tr],y[tr],rcond=None)[0]
    def link(z,a):
        if kind=='tanh': return np.tanh(a*z)/a
        if kind=='atan': return np.arctan(a*z)/a
        if kind=='sin': return np.sin(a*z)/a
        if kind=='rational': return z/(1+(a*z)**2)
        if kind=='erf': return erf(a*z)/a*np.sqrt(np.pi)/2
    def predict(p,i):
        z=c[i]@p[1:5]
        if mode=='outside': return p[0]+p[5]*old[i]+link(z,p[6])
        if mode=='shift': return p[0]+p[5]*old[i]+link(z+p[7],p[6])
        if mode=='mixed': return p[0]+p[5]*old[i]+z+p[7]*(link(z,p[6])-z)
        if mode=='mixed_offset': return p[0]+p[5]*old[i]+z+p[7]*(link(z+p[8],p[6])-(z+p[8]))
        if mode=='separate': return p[0]+p[5]*old[i]+z+(link(c[i]@p[7:11],p[6])-c[i]@p[7:11])
        if mode=='interactions': return p[0]+p[5]*old[i]+link(z+p[7]*old[i]+p[8],p[6])
        if mode=='scaled_bilinear': return p[0]+p[5]*old[i]+link(z,p[6])+p[7]*old[i]*z+p[8]*old[i]*z*z
    extra={'outside':[], 'shift':[0.], 'mixed':[1.], 'mixed_offset':[1.,0.], 'separate':list(b[1:5]), 'interactions':[0.,0.], 'scaled_bilinear':[0.,0.]}[mode]
    x=np.r_[b,.35,extra]; lo=np.full(len(x),-np.inf); hi=-lo; lo[6]=.015; hi[6]=1.5
    if mode.startswith('mixed'): lo[7]=0.; hi[7]=5.
    opt=least_squares(lambda p:predict(p,tr)-y[tr],x,bounds=(lo,hi),max_nfev=200)
    return np.mean(np.abs(predict(opt.x,te)-y[te])),opt.x

scores=[]
for kind in ('tanh','atan','sin','rational','erf'):
    for mode in ('outside','shift','mixed','mixed_offset','separate','interactions','scaled_bilinear'):
        err=np.array([evaluate(kind,mode,tr,te)[0] for tr,te in folds]); scores.append((err[1:].mean(),kind,mode,err))
for _,kind,mode,e in sorted(scores)[:16]:
    print(kind,mode,'holdout/oof/worst',np.round([e[0],e[1:].mean(),e.max()],6),'full fit',evaluate(kind,mode,np.arange(len(y)),np.arange(len(y)))[1].round(5))

from rheology_probe import D, B, t, design
import numpy as np
from scipy.optimize import least_squares
train=D['train']; pilot=D['chamber_verification_trials']; test=D['test']

def panel(pool):
 g=np.concatenate([a['lab_id'] for a in pool]); r=np.concatenate([a['ref_coupons'] for a in pool]); return {lab:(np.median(r[g==lab].mean(0)),np.median(r[g==lab].std(0))) for lab in np.unique(g)}

def prep(a,ref,fill=None):
 g=a['lab_id']; s=a['stratum']; x=a['coupon_features_raw'].copy(); _,i,n=np.unique(np.c_[g,s],axis=0,return_counts=True,return_inverse=True); w=1/n[i]; w/=w.mean()
 for lab in np.unique(g):
  j=g==lab; loc,sd=ref[lab]; x[j]=(x[j]-loc)/sd
 m=np.isfinite(x)
 if fill is None: fill=(np.where(m,x,0)*w[:,None]).sum(0)/(m*w[:,None]).sum(0)
 x=np.where(m,x,fill); p=a['past_channel_raw']@B[1:].sum(1)+B[0].sum(); h=a['hold_channel_raw']@B[1:].sum(1)+B[0].sum()
 y=np.c_[a['instant_rate_obs'],a['strain_curves_obs']-a['strain_0'][:,None]]; return (p,h,np.c_[np.ones(len(g)),x],y,w),fill

def sub(a,idx): return tuple(x[idx] for x in a)

def fit(parts,init=None):
 def resid(v):
  tau=np.sort(np.exp(v[:2])); return np.concatenate([((design(a,tau)@v[2:]-a[3])*np.sqrt(a[4][:,None])/sig).ravel() for a,sig in parts])
 a=parts[0][0]; sols=[]
 for q in ([.65,4],[.25,1.4],[1.2,12]) if init is None else [init]:
  c=np.linalg.lstsq(design(a,np.array(q)).reshape(-1,7),a[3].ravel(),rcond=None)[0]; z=least_squares(resid,np.r_[np.log(q),c],bounds=(np.r_[[np.log(.05001)]*2,[-np.inf]*7],np.r_[[np.log(24.999)]*2,[np.inf]*7]),loss='cauchy',max_nfev=140); sols.append(z)
 z=min(sols,key=lambda z:z.cost); return np.sort(np.exp(z.x[:2])),z.x[2:]

def err(a,fit): return (design(a,fit[0])@fit[1]-a[3])

def report(e): return np.array([np.median(abs(e[:,0])),np.median(abs(e[:,1:])),np.max(np.median(abs(e[:,1:]),axis=1)),np.median(abs(e[:,-1]))]).round(5)

if __name__=='__main__':
 for refname,pool in [('own',[pilot]),('train',[train]),('pool',[train,pilot]),('all',[train,pilot,test])]:
  ref=panel(pool); a,f=prep(pilot,ref); atr,ft=prep(train,panel([train])); print('reference',refname,'pilot observed features',np.isfinite(pilot['coupon_features_raw']).sum(axis=0),'fill',f.round(3))
  for label,kind,st,weight in [('train',0,.025,1),('pilot',1,.005,1),('pilot tight',1,.002,1),('pool',2,.005,1),('pool downweight train',2,.005,.2)]:
   ep=np.empty_like(a[3]); estimates=[]
   for lab in np.unique(pilot['lab_id']):
    j=pilot['lab_id']==lab; v=sub(a,~j); tr=tuple(list(atr[:4])+[atr[4]*weight]); parts=[(tr,np.array([.012]+[.027]*(len(t)-1)))] if kind!=1 else []
    if kind: parts.insert(0,(v,np.array([st/2]+[st]*(len(t)-1))))
    z=fit(parts); ep[j]=err(sub(a,j),z); estimates.append(z[0].round(3).tolist())
   print(label,'OOF pooled',report(ep),'worst lab medians',np.max([report(ep[pilot['lab_id']==g]) for g in np.unique(pilot['lab_id'])],axis=0),'tau',estimates)

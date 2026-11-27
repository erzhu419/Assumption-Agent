from chamber_probe import *
from itertools import combinations
from scipy.optimize import least_squares
past,H,F=inp_d; lab=d['lab_id']; s=d['stratum']; miss=np.isnan(d['coupon_features_raw']).any(axis=1)
U=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); T=np.r_[t,1.]; R=np.c_[yd,rd]-H[:,None]*(U@th[:2]); _,G=m._design(tau,inp_d,dt,t); Z=H[:,None]-G[:,:2]*tau
x=F[:,1:]; one=F[:,0]; r=F@th[2:]; raw=d['coupon_features_raw'].copy(); raw=np.where(np.isfinite(raw),raw,np.nanmean(raw,axis=0));
for g in np.unique(lab):
 print('lab',g,[(np.median(v['ref_coupons'][v['lab_id']==g].mean(axis=0)).round(4),np.median(v['ref_coupons'][v['lab_id']==g].std(axis=0)).round(4)) for v in (tr,te,d)])
print('diag keys',list(d)); print('test keys',list(te));
def predict(keep, cols):
 # Fit the temporal mode and amplitude entirely within the fold.
 _,_,v=np.linalg.svd(R[keep],full_matrices=False); prof=v[0]/v[0,-1]; B=np.c_[U,T]; prof=B@np.linalg.lstsq(B,prof,rcond=None)[0]; prof/=prof[-1]; a=R[keep]@prof/(prof@prof); c=np.linalg.lstsq(cols[keep],a,rcond=None)[0]; return (cols@c)[:,None]*prof,c
base=np.c_[one,r]; models={'r':base,'x':F,'raw':np.c_[one,raw],'z':np.c_[base,Z],'r*z1':np.c_[base,r*Z[:,1]],'r*past4':np.c_[base,r*past[:,4]],'r*z1^2':np.c_[base,r*Z[:,1]**2],'r*z1+z1':np.c_[base,Z[:,1],r*Z[:,1]],'r*z0,z1':np.c_[base,Z,r[:,None]*Z]}
# Test a scalar regularized chamber amplitude and simple feature interactions.
for i,j in combinations(range(4),2):
 for l,a in [('prod',x[:,i]*x[:,j]),('sqdiff',(x[:,i]-x[:,j])**2),('sqsum',(x[:,i]+x[:,j])**2),('rational',r/(1+(x[:,i]-x[:,j])**2))]: models[f'{l}{i}{j}']=np.c_[base,a]
for v,name in ((H,'H'),(Z[:,0],'z0'),(Z[:,1],'z1'),(Z[:,0]-Z[:,1],'z0-z1')):
 for a in (.1,.25,.5,1.,2.,4.): models[f'r/(1+{a}*{name}^2)']=np.c_[one,r/(1+a*v*v)]
rank=[]
for name,M in models.items():
 E=np.zeros_like(R)
 for g in np.unique(lab):
  k=lab!=g; E[~k]=R[~k]-predict(k,M)[0][~k]
 worst=np.max([np.sqrt(np.mean(E[lab==g]**2,axis=0)) for g in np.unique(lab)],axis=0)
 err=R-predict(np.ones(len(R),bool),M)[0]; rank.append((max(worst[:-1]),worst[-1],name,score(err),predict(np.ones(len(R),bool),M)[1].round(5)))
print(*sorted(rank)[:16],sep='\n');print('reference',*[a for a in rank if a[2] in ('r','x','raw','r*z1')],sep='\n')

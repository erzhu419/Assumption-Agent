import pickle
import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.linalg import qr
from prony_probe import memory, fit
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment']; t,x,y,tt,xx=[e[k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain')]; truth=p['gt_test_stress']; D=np.c_[np.ones(len(t)),t/t[-1]]; taus=np.geomspace(.02,20,90)
def design(t,x,ts): return np.c_[x,memory(t,x,ts)]
A=design(t,x,taus)
c,b,r,s=fit(A,D,y,True)
keep=np.abs(r-np.median(r))<7*s
ix=np.flatnonzero(c[1:]>1e-7); groups=np.split(ix,np.where(np.diff(ix)>2)[0]+1); z0=np.array([np.average(np.log(taus[g]),weights=c[1:][g]) for g in groups]); m=len(z0)
def solve(B,D,y):
    Q=qr(D,mode='economic')[0]; C=B-Q@(Q.T@B); yy=y-Q@(Q.T@y); scale=np.linalg.norm(C,axis=0); cc=nnls(C/scale,yy,maxiter=5000)[0]/scale; bb=np.linalg.lstsq(D,y-B@cc,rcond=None)[0]; return cc,bb,y-B@cc-D@bb

def opt(y,train,penalty=0,z0=z0):
    d=D[train]; yy=y[train]
    def calc(z):
        B=design(t,x,np.exp(z)); return B,solve(B[train],d,yy)
    def fun(z): return np.r_[calc(z)[1][2],s*penalty*np.diff(z,n=2)]
    sol=least_squares(fun,z0,bounds=(np.log(.02),np.log(20.)),max_nfev=120,ftol=1e-9); B,(c,b,r)=calc(sol.x); return sol.x,c,b,r,B

def error(c,z):
    pp=design(tt,xx,np.exp(z))@c
    return np.linalg.norm(pp-truth)/np.linalg.norm(truth),max(np.linalg.norm((pp-truth)[q])/np.linalg.norm(truth[q]) for q in np.array_split(np.arange(len(tt)),4))
for penalty in (0,.4,1,2,4,8,16,64):
    z,c,b,r,B=opt(y,keep,penalty); cv=[]; g=[]
    # Buffered contiguous validation windows, spread over ramp, hold, and sinusoidal phases.
    windows=[(0,14),(23,37),(53,82),(126,157),(203,234),(271,299),(339,373),(421,453),(493,525),(558,587)]
    for a,h in windows:
        va=keep & (np.arange(len(t))>=a) & (np.arange(len(t))<h); tr=keep & ~((np.arange(len(t))>=max(0,a-2)) & (np.arange(len(t))<h+2)); zz,cc,bb,_,BB=opt(y,tr,penalty); rr=(y-BB@cc-D@bb)[va]; cv.append(np.sqrt(np.mean(rr**2))); g.append(cc[0]);
    print('penalty',penalty,'G',round(c[0],5),'tau',np.exp(z).round(4),'drift',b.round(5),'noise',np.sqrt(np.mean(r*r)),'test',error(c,z),'CV rms/max',np.sqrt(np.mean(np.square(cv))),max(cv),'CV G range',min(g),max(g))
# Expected modulus uncertainty without access to labels: noise-scale movement of the slow mode.
z,c,b,r,B=opt(y,keep,0)
for slow in (7,8,9,10,11,12,14,16,20):
    def fun(v):
        B=design(t,x,np.r_[np.exp(v),slow]); return solve(B[keep],D[keep],y[keep])[2]
    v=least_squares(fun,z[:-1],bounds=(np.log(.02),np.log(20.)),max_nfev=90).x
    ts=np.r_[np.exp(v),slow]; cc,bb,rr=solve(design(t,x,ts)[keep],D[keep],y[keep]); print('profile',slow,'G',round(cc[0],3),'delta-chi2',round((rr@rr-r@r)/s**2,3),'stress',error(cc,np.log(ts)))

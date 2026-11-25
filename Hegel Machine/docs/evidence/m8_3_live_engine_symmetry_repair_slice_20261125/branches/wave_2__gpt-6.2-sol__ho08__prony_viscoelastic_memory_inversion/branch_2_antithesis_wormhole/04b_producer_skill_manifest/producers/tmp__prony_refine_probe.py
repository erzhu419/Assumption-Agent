import pickle
import numpy as np
from scipy.linalg import qr
from scipy.optimize import least_squares, nnls
from prony_probe import memory
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment']
t,x,y,tt,xx=[e[k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain')]; yt=p['gt_test_stress']; D=np.column_stack((np.ones(len(t)),t/t[-1])); tau=np.geomspace(.02,20,100)
A=np.column_stack((x,memory(t,x,tau))); At=np.column_stack((xx,memory(tt,xx,tau)))
def solve(A,w):
    Aw=A*w[:,None]; Dw=D*w[:,None]; yw=y*w
    Q=qr(Dw,mode='economic')[0]; Ap=Aw-Q@(Q.T@Aw); yp=yw-Q@(Q.T@yw)
    s=np.linalg.norm(Ap,axis=0); c=nnls(Ap/s,yp,maxiter=5000)[0]/s
    b=np.linalg.lstsq(Dw,(y-A@c)*w,rcond=None)[0]
    return c,b,y-A@c-D@b

def report(label,c,b,r,pred,taus):
    mad=1.4826*np.median(np.abs(r-np.median(r)))
    err=[np.linalg.norm(pred[i]-yt[i])/np.linalg.norm(yt[i]) for i in np.array_split(np.arange(len(yt)),4)]
    print(label,'modulus %.5f'%c[0],'mod_error %.5f'%abs(c[0]/p['gt_G_inf']-1),'stress %.6f'%(np.linalg.norm(pred-yt)/np.linalg.norm(yt)),'worst %.6f'%max(err),'mad %.5f'%mad,'trim',np.sum(abs(r)>5*mad),'drift',b.round(4),'modes',list(zip(taus[c[1:]>1e-7].round(4),c[1:][c[1:]>1e-7].round(4))))

for width in (1.,2.4,4.,8.):
    w=np.ones(len(y))
    for _ in range(22):
        c,b,r=solve(A,w); mad=1.4826*np.median(np.abs(r-np.median(r))); w=1/np.hypot(1,r/(width*mad))
    report('grid cauchy '+str(width),c,b,r,At@c,tau)
    if width==2.4: base=c,b,r,mad
c,b,r,mad=base
print('largest residuals by magnitude:',[(round(t[i],2),round(r[i],5)) for i in np.argsort(abs(r))[-24:][::-1]])
inds=np.flatnonzero(c[1:]>1e-7); groups=np.split(inds,np.where(np.diff(inds)>1)[0]+1)
seed_t=np.array([np.average(np.log(tau[g]),weights=c[1:][g]) for g in groups]); seed_c=np.r_[c[0],[np.sum(c[1:][g]) for g in groups]]
for thresh in (3.5,5.,8.):
    w=(abs(r)<=thresh*mad).astype(float); cc,bb,rr=solve(A,w)
    report('grid trim '+str(thresh),cc,bb,rr,At@cc,tau)
for model in ('cauchy','trim3.5','trim5','trim8','huber'):
    m=len(groups); v0=np.r_[seed_t,seed_c,b]
    bounds=(np.r_[np.repeat(np.log(.02),m),np.zeros(m+1),-np.inf,-np.inf],np.r_[np.repeat(np.log(20.),m),np.repeat(np.inf,m+3)])
    keep=abs(r)<=float(model[4:])*mad if model.startswith('trim') else np.ones(len(r),bool)
    def resid(v,obs=None):
        B=np.column_stack((x,memory(t,x,np.exp(v[:m])))); pred=B@v[m:2*m+1]+D@v[-2:]
        return (pred-y)[keep] if obs is None else y-pred
    opt=least_squares(resid,v0,bounds=bounds,loss='linear' if model.startswith('trim') else model,f_scale=2.4*mad,x_scale='jac',max_nfev=200)
    v=opt.x; ts=np.exp(v[:m]); cc=v[m:2*m+1]; rt=resid(v,True); pred=np.column_stack((xx,memory(tt,xx,ts)))@cc
    report('free '+model,cc,v[-2:],rt,pred,ts)
    if model=='trim5':
        _,s,Vt=np.linalg.svd(opt.jac,full_matrices=False); cov=(Vt.T/s**2)@Vt*mad**2
        print('trim5 parameter stderr',np.sqrt(np.diag(cov)).round(3),'correlation of G_inf with other params',(cov[m]/np.sqrt(cov[m,m]*np.diag(cov))).round(3))

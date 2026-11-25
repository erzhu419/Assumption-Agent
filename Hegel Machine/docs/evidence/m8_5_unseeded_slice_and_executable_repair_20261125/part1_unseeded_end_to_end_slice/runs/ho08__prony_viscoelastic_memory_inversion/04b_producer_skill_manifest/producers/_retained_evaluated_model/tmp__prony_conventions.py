import numpy as np,pickle
from scipy.signal import lfilter
from scipy.optimize import nnls,least_squares
from prony_probe import solve

def build(t,e,tau,kind):
    dt=np.diff(t)[:,None]; dx=np.diff(e)[:,None]; x=dt/tau
    if kind=='euler':
        a=1-x; c=1
    elif kind=='backward_euler':
        a=1/(1+x); c=a
    else:
        a=np.exp(-x)
        c={'linear':lambda:-np.expm1(-x)/x,'right':lambda:1,'left':lambda:a,'mid':lambda:np.exp(-x/2),'trapezoid':lambda:(1+a)/2}[kind]()
    q=np.zeros((len(t),len(tau))); f=c*dx*np.ones_like(a)
    if np.allclose(dt,dt[0],atol=0,rtol=1e-10):
        for k,v in enumerate(a[0]): q[1:,k]=lfilter([1.],[1.,-v],f[:,k])
    else:
        for k in range(len(dt)): q[k+1]=a[k]*q[k]+f[k]
    return np.column_stack((e,q))

if __name__=='__main__':
    p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment']; t=e['train_t']; eps=e['train_strain']; y=e['train_stress_raw']; te=e['test_t']; et=e['test_strain']; yt=p['gt_test_stress']; G=p['gt_G_inf']
    D=np.c_[np.ones_like(t),(t-t.mean())/np.ptp(t)]; nrm=np.linalg.norm(yt)
    for kind in ['linear','right','left','mid','trapezoid','euler','backward_euler']:
        tau=np.geomspace(max(.02,.051 if kind=='euler' else .02),20,81); A=build(t,eps,tau,kind); B=build(te,et,tau,kind)
        g,b,s,r=solve(A,D,y,'clip')
        # A separate representation diagnostic on the public clean trace, never used to estimate the production model.
        g0=nnls(B[:,1:],yt-G*et,atol=1e-13,maxiter=3000)[0]; active=g0>1e-7; tt=tau[active]; gg=g0[active]
        res=least_squares(lambda v: (build(te,et,np.exp(v[:len(tt)]),kind)[:,1:]@v[len(tt):]+G*et-yt),np.r_[np.log(tt),gg],bounds=(np.r_[[np.log(tau[0])]*len(tt),[0]*len(tt)],np.r_[[np.log(20)]*len(tt),[np.inf]*len(tt)]),max_nfev=100,gtol=1e-12)
        print(kind,'train G',g[0],'test rel',np.linalg.norm(B@g-yt)/nrm,'noise',s,'fixed-G clean representation error',np.linalg.norm(res.fun)/nrm,'modes',list(zip(np.exp(res.x[:len(tt)]).round(5),res.x[len(tt):].round(5))))
    r=y-build(t,eps,np.geomspace(.02,20,61),'linear')@solve(build(t,eps,np.geomspace(.02,20,61),'linear'),D,y,'clip')[0]; r-=D@solve(build(t,eps,np.geomspace(.02,20,61),'linear'),D,y,'clip')[1]
    for lo,hi in [(0,2),(2,14),(14,15),(15,30.1)]:
        z=r[(t>=lo)&(t<hi)]; z=z[abs(z)<.07]; print((lo,hi),np.mean(z),np.std(z),len(z))

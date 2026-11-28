import pickle
import numpy as np
from scipy.optimize import least_squares

study = pickle.load(open('public_batch.pkl','rb'))[0]['study']
x, y = (study['panel_b'][key] for key in ('u','v'))
forms = {
    'rational': lambda x,k: x**3/(1+k*x*x),
    'tanh_residual': lambda x,k: x-np.tanh(k*x)/k,
    'sin_residual': lambda x,k: x-np.sin(k*x)/k,
    'atan_residual': lambda x,k: x-np.arctan(k*x)/k,
    'exp_saturation': lambda x,k: x*(-np.expm1(-k*x*x)),
    'asinh_residual': lambda x,k: x-np.arcsinh(k*x)/k,
    'softsign_residual': lambda x,k: x*(1-1/np.sqrt(1+k*x*x)),
}
def design(x,k,form,aff=False):
    z=form(x,k)
    return np.column_stack((np.ones(len(x)),x,z)) if aff else np.column_stack((np.ones(len(x)),z))
def profile(x,y,form,aff=False,split=False,start=1.,top=5.):
    def calc(k):
        A=design(x,k,form,aff)
        c=np.linalg.lstsq(A,y,rcond=None)[0]
        return y-A@c,c
    sol=least_squares(lambda k:calc(k)[0].ravel(),np.full(4 if split else 1,start),bounds=(0.005,top),max_nfev=150,gtol=1e-12)
    return sol.x, *calc(sol.x)
for name,form in forms.items():
    for aff,split in ((False,False),(True,False),(False,True),(True,True)):
        seeds=(.2,1.,2.7) if name=='sin_residual' else (.8,)
        def choose(a,b):
            return min((profile(a,b,form,aff,split,k) for k in seeds),key=lambda v:np.linalg.norm(v[1]))
        k,e,c=choose(x,y)
        errs=[]
        for i in range(len(x)):
            mask=np.arange(len(x))!=i
            kk,_,cc=choose(x[mask],y[mask]); errs.append(y[i]-design(x[i:i+1],kk,form,aff)@cc)
        cv=np.sqrt(np.mean(np.vstack(errs)**2,axis=0))
        print(name,aff,split,'k',np.round(k,5),'fit',round(np.sqrt(np.mean(e**2)),7),'loo',np.round(cv,6), 'maxres',round(np.max(np.abs(e)),6))
        if np.max(cv)<0.01: print('  coefficients',np.round(c,5).tolist())

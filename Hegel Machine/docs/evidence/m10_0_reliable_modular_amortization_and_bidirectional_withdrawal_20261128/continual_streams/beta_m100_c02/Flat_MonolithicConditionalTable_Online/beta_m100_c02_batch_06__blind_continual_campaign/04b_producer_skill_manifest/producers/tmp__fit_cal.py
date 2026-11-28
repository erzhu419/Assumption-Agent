import numpy as np
from scipy.optimize import least_squares
from numpy.polynomial.chebyshev import chebvander

def fit_cal(sweeps, held_out=(), family='tanh', degree=8):
    X,Y=(np.asarray(sweeps[k]) for k in ('ref_rosette','raw_rosette'))
    s=sweeps['single_bank_sweeps']; models=[]
    for j in range(4):
        sel=s['bank_id']==j
        x=np.r_[s['ref_signal'][sel],X[:,j]]; z=np.r_[np.zeros(sel.sum()),X[:,(j+2)%4]]
        y=np.r_[s['raw_readout'][sel],Y[:,j]]; keep=abs(y)<5
        keep[sel.sum()+np.asarray(held_out,dtype=int)]=False
        x,z,y=x[keep],z[keep],y[keep]
        def basis(t):
            return (chebvander(t/1.8,degree)-chebvander(0.,degree))[:,1:]
        B=basis(z)
        if family=='poly':
            A=np.c_[np.ones(len(x)),basis(x)[:,::2],B]; p=np.linalg.lstsq(A,y,rcond=None)[0]
            def poly(t,u,p=p,basis=basis,n=1+basis(x).shape[1]//2):
                t,u=np.asarray(t),np.asarray(u); return p[0]+basis(t)[:,::2]@p[1:n]+basis(u)@p[n:]
            models.append(poly);continue
        def odd(t,b, family=family):
            if family=='tanh': return np.tanh(b*t)
            if family=='atan': return np.arctan(b*t)
            if family=='rational': return (b*t)/np.sqrt(1+(b*t)**2)
        def design(b):return np.c_[np.ones(len(x)),x,odd(x,b),B]
        def err(p):return design(p[0])@p[1:]-y
        b=least_squares(lambda q: design(q[0])@np.linalg.lstsq(design(q[0]),y,rcond=None)[0]-y,[1.3],bounds=(0.1,8.)).x[0]
        p=np.r_[b,np.linalg.lstsq(design(b),y,rcond=None)[0]]
        p=least_squares(err,p,loss='soft_l1',f_scale=3e-5,gtol=1e-11,max_nfev=100).x
        def fun(t,u,p=p,odd=odd,basis=basis):return p[1]+p[2]*t+p[3]*odd(t,p[0])+basis(u)@p[4:]
        fun.par=p;models.append(fun)
    return models

if __name__=='__main__':
    import pickle
    s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; ok=abs(Y)<5
    for family,degree in [('tanh',4),('tanh',6),('tanh',8),('tanh',10),('atan',8),('rational',8),('poly',8),('poly',10)]:
        e=np.empty_like(Y)
        for fold in range(5):
            i=np.arange(fold,len(X),5); mm=fit_cal(c,i,family,degree)
            for b in range(4):e[i,b]=mm[b](X[i,b],X[i,(b+2)%4])-Y[i,b]
        print(family,degree,'OOF', [round(np.sqrt(np.mean(e[ok[:,j],j]**2)),8) for j in range(4)],'max',abs(e[ok]).max())
        if family!='poly' and degree==8: print('parameters',np.round([m.par for m in fit_cal(c,(),family,degree)],6))

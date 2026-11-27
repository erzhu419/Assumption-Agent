import importlib.util as iu
import pickle
import numpy as np
from scipy.optimize import least_squares

p='/holdout_task/'
def load(name,path):
    spec=iu.spec_from_file_location(name,path)
    m=iu.module_from_spec(spec); spec.loader.exec_module(m); return m
m=load('candidate',p+'batch_solver.py')
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
b=s['sensor_calibration_sweeps']; R=b['ref_rosette']; Y=b['raw_rosette']; single=b['single_bank_sweeps']
for j in range(4):
    same=single['bank_id']==j; alone=np.zeros((same.sum(),4)); alone[:,j]=single['ref_signal'][same]
    r=np.r_[R,alone]; t=np.r_[Y[:,j],single['raw_readout'][same]]
    x,n,q,o=r[:,j],r[:,(j+1)%4],r[:,(j-1)%4],r[:,(j+2)%4]
    D=np.c_[np.ones(len(t)),x,n,q,x*n,x**3]; Q=np.c_[x*x,n*n,(x-n)**2,(x+q)**2]
    err=[]
    for f in range(5):
        test=np.arange(len(t))%5==f; train=~test
        clean=train & (np.abs(t)<8); valid=test & (np.abs(t)<8)
        c=np.linalg.lstsq(D[clean],t[clean],rcond=None)[0]; base=D[valid]@c-t[valid]
        def pred(h, mask): return (D[mask]@h[:6])/(1+Q[mask]@h[6:])
        h=least_squares(lambda h:pred(h,clean)-t[clean],np.r_[c,np.full(4,1e-8)],bounds=(np.r_[np.full(6,-np.inf),np.zeros(4)],np.full(10,np.inf)),gtol=1e-13,ftol=1e-12,xtol=1e-12).x
        err.append([np.mean(base**2)**.5,np.mean((pred(h,valid)-t[valid])**2)**.5,max(h[6:])])
    print('calibration bank',j,'worst held-out polynomial/rational RMSE, max quotient coefficient:',np.max(err,axis=0))
S,U=m._forward(s); y=m._calibrated_observations(s); Z=s['Z_nuisance']; ridge=s['source_ridge']; eta=s['nuisance_ridge']; cap=s['u_box']; budget=s['total_budget']; I=np.eye(S.shape[1]); IZ=np.eye(Z.shape[1]); ns=len(s['sensor_ij']); nk=len(s['mu_shifts'])
def fit(A,B,t,nuisance):
    joined=np.c_[A,t]; C=np.linalg.solve(B.T@B+eta*IZ,B.T@joined) if nuisance else np.zeros((B.shape[1],joined.shape[1])); d=joined-B@C; D,r=d[:,:-1],d[:,-1]
    w,nu=m._box_simplex(D.T@D+ridge*I,-D.T@r,cap,budget)
    return w,nu,C[:,-1]-C[:,:-1]@w,D,r
for grouping in [np.tile(np.arange(ns),nk),np.repeat(np.arange(nk),ns)]:
    scores=[]
    for f in np.unique(grouping):
        te=grouping==f; tr=~te; r=[]
        for nuisance in [True,False]:
            w,_,g,_,_=fit(S[tr],Z[tr],y[tr],nuisance)
            e=S[te]@w+Z[te]@g-y[te]; r.append(np.mean(e*e)**.5)
        scores.append(r)
    print('worst sensor/shift held-out RMSE with/without nuisance:',np.max(scores,axis=0))
a=m.solve_batch(s); w=np.asarray(a['weights']); c=a['kkt_certificate']; nu=c['nu_sum']; _,_,_,D,r=fit(S,Z,y,True); e=D@w-r; g=D.T@e+ridge*w
J=(e@e+ridge*(w@w))/2; lo=np.asarray(c['lam_lower']); hi=np.asarray(c['lam_upper']); dual=J-g@w+nu*budget+cap*np.minimum(0,g-nu).sum(); field=np.tensordot(w,U,axes=(0,0))
print('budget error; box violation; stationarity infinity norm; complementarity infinity norm; duality gap:',abs(w.sum()-budget),max(0,-w.min(),w.max()-cap),np.max(abs(g-nu-lo+hi)),max(np.max(abs(lo*w)),np.max(abs(hi*(cap-w)))),J-dual)
np.testing.assert_allclose(c['weights'],w,rtol=0,atol=1e-12)
np.testing.assert_allclose([c['primal_objective'],c['dual_lower_bound']],[J,dual],rtol=1e-10,atol=1e-12)
np.testing.assert_allclose(a['terminal_field'],field,rtol=1e-12,atol=1e-12)
assert abs(w.sum()-budget)<1e-8 and min(w.min(),cap-w.max())>=-1e-8
assert min(lo.min(),hi.min())>=0 and np.max(abs(g-nu-lo+hi))<1e-9

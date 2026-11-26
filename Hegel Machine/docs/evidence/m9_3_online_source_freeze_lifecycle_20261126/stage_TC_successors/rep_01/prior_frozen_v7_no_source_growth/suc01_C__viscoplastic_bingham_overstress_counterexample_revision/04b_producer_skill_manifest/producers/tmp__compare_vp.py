import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, hstack
p='/holdout_task/viscoplastic_overstress_hereditary_solver.py'
s=importlib.util.spec_from_file_location('baseline',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

def joint(design, train, curve_sigma=0.025, onset_sigma=0.016, null=False):
    taus, coeff = m._fit(design, train)
    t = design.time
    y = train['strain_curves_obs'] - train['strain_0'][:,None]
    onset = train['instant_rate_obs']
    n,k = y.shape
    idx = np.array([6,8] if null else [5,6,7,8])
    def un(q):
        return np.array([q[0],q[0]+(8-q[0])*q[1]])
    def evaluate(q):
        c,r = design.matrices(un(q))
        return c[:,:,idx]@q[2:],r[:,idx]@q[2:]
    def physical(q):
        c,r = evaluate(q)
        return np.r_[(c/curve_sigma).ravel(),r/onset_sigma]
    a = csr_matrix((np.r_[np.tile(t/curve_sigma,n), np.full(n,1/onset_sigma)], (np.arange(n*(k+1)),np.r_[np.repeat(np.arange(n),k),np.arange(n)])), shape=(n*(k+1),n))
    obs = np.r_[(y/curve_sigma).ravel(),onset/onset_sigma]
    q0 = np.r_[taus[0],(taus[1]-taus[0])/(8-taus[0]),coeff[idx]]
    _,r = evaluate(q0)
    nd = len(q0)
    def fun(q): return physical(q[:nd])+a@q[nd:]-obs
    def jac(q):
        q = q[:nd]
        c,r = design.matrices(un(q)); b = np.empty((len(obs),nd)); b[:,2:] = np.vstack(((c[:,:,idx]/curve_sigma).reshape(-1,len(idx)),r[:,idx]/onset_sigma))
        for j in range(2):
            delta = np.zeros(nd); delta[j]=1e-5
            b[:,j]=(physical(q+delta)-physical(q-delta))/(2e-5)
        return hstack((csr_matrix(b),a),format='csr')
    def loss(z):
        result = np.empty((3,len(z))); c = z[:n*k]/9
        result[:, :n*k] = np.stack((9*np.log1p(c),1/(1+c),-1/(9*(1+c)**2)))
        result[:,n*k:] = np.stack((z[n*k:],np.ones(n),np.zeros(n)))
        return result
    fit = least_squares(fun,np.r_[q0,onset-r],jac=jac,loss=loss,bounds=(np.r_[.2,1e-6,np.full(nd+n-2,-np.inf)],np.r_[8-1e-6,1,np.full(nd+n-2,np.inf)]),x_scale='jac',max_nfev=100,tr_options={'atol':1e-9,'btol':1e-9})
    coef = np.zeros(9); coef[idx] = fit.x[2:nd]
    rr=fun(fit.x); print('noise',np.median(np.abs(rr[:n*k]))*curve_sigma,np.median(np.abs(rr[n*k:]))*onset_sigma,end='; ')
    return un(fit.x),coef

def subset(d, mask): return {k:v[mask] for k,v in d.items()}
pub=pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl','rb'))[0]; d,g=pub['dataset'],pub['gt']; t,dt,y=d['eval_times'],d['past_dt'],d['yield_stress']
rng=np.random.default_rng(902); tr=d['train']; n=len(tr['lab_id']); fold=np.zeros(n,int)
for l in range(4):
    for j in range(3):
        inds=np.flatnonzero((tr['lab_id']==l)&(tr['stratum']==j)); fold[rng.permutation(inds)]=np.arange(len(inds))%4
for split in range(-1,4):
    train,test=(tr,d['test']) if split<0 else (subset(tr,fold!=split),subset(tr,fold==split)); x,xt=m._align(train,test)
    D=m._Design(train,x,dt,t,y); T=m._Design(test,xt,dt,t,y)
    truth=g['te_clean'] if split<0 else test['strain_curves_obs']; over=(np.max(abs(test['past_stress']),1)>y)|(abs(test['stress_hold_0'])>y)
    for name,fit in [('base',m._fit),('joint',joint),('joint2',lambda a,b:joint(a,b,.019,.019)),('null',lambda a,b:joint(a,b,null=True))]:
        taus,coef=fit(D,train); c,r=T.matrices(taus); pred=test['strain_0'][:,None]+c@coef+(test['instant_rate_obs']-r@coef)[:,None]*t
        err=abs(pred-truth); metric=np.mean if split<0 else lambda r: np.mean(np.minimum(r,.15)); print(split,name,taus.round(5),coef[[5,7]].round(5), 'slice err',*[round(metric(err[mask]),5) for mask in (over|~over,over,~over)],'early/late',np.round([metric(err[:,:3]),metric(err[:,-3:])],5))

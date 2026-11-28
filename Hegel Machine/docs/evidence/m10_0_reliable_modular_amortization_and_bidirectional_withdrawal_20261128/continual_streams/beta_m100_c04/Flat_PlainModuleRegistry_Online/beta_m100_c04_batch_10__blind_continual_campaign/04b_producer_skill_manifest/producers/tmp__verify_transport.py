import sys,pickle,json,itertools
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.optimize import minimize
from batch_solver import solve_batch
from downstream_solver import solve_downstream
from _verified_upstream import calibrate_upstream
p='/holdout_task/'
items=pickle.load(open(p+'public_batch.pkl','rb'))
pay=json.load(open(p+'_upstream_param_payload.json'))

def axis(k,v,h,wall):
    n=len(k); a=np.zeros((n,n))
    for i in range(n):
        if wall=='sealed' and i in (0,n-1):
            j=1 if i==0 else n-2
            a[i,j]=-k[i]/h**2
        else:
            for step in (-1,1):
                j=(i+step)%n
                a[i,j]-=(k[i]+k[j])/(2*h**2)+max(-step*v[i],0)/h
        a[i,i]=-a[i].sum()
    return a

for item in items:
    study=item['study']; c=calibrate_upstream(study); y=c['y_obs_raw']; z=c['Z_nuisance']
    out=solve_batch(study); dn=solve_downstream(c)
    for key in ('weights','terminal_field'):
        np.testing.assert_array_equal(out[key],dn[key]); assert np.isfinite(out[key]).all()
    assert out['kkt_certificate']==dn['kkt_certificate']
    print('calibration payload vs sweep-fit, max deviation',np.max(np.abs(y-calibrate_upstream(study,pay)['y_obs_raw'])))
    nx,ny=c['nx'],c['ny']; n=nx*ny
    rhs=np.stack([a.ravel(order='F') for a in c['source_modes']],axis=1)
    rows=np.ravel_multi_index(c['sensor_ij'].T,(nx,ny),order='F')
    base=np.diag(c['V_reaction_2d'].ravel(order='F')); V=np.eye(n)
    cap=c['u_box']; budget=c['total_budget']; w=out['weights']; cert=out['kkt_certificate']; ridge=c['source_ridge']
    starts=np.full(len(w),budget/len(w)); bounds=[(0,cap)]*len(w)
    eq={'type':'eq','fun':lambda u:u.sum()-budget,'jac':lambda u:np.ones_like(u)}
    groups=np.tile(np.arange(len(rows))//4,len(c['mu_shifts']))
    for walls in itertools.product(('sealed','periodic'),repeat=2):
        a,b=[axis(c['kappa_'+k],c['vel_'+k],c['L'+k]/(size-1),wall) for k,size,wall in zip('xy',(nx,ny),walls)]
        L=base+np.kron(np.eye(ny),a)+np.kron(b,np.eye(nx))
        fields=[np.linalg.solve(L+mu*V,rhs) for mu in c['mu_shifts']]
        assert max(np.max(np.abs((L+mu*V)@F-rhs)) for mu,F in zip(c['mu_shifts'],fields))<1e-9
        S=np.vstack([F[rows] for F in fields]); D=np.column_stack((S,z)); pen=np.r_[np.zeros(len(w)),np.full(z.shape[1],np.inf)]
        def empirical(mask):
            X=S[mask]; Z=z[mask]; Y=y[mask]; K=np.linalg.lstsq(Z,np.c_[X,Y],rcond=None)[0]; T=X-Z@K[:,:-1]; R=Y-Z@K[:,-1]
            fit=minimize(lambda u:0.5*np.sum((T@u-R)**2),starts,jac=lambda u:T.T@(T@u-R),bounds=bounds,constraints=eq,method='SLSQP',options={'ftol':1e-15,'maxiter':250})
            assert fit.success, fit.message
            return y-S@fit.x-z@(K[:,-1]-K[:,:-1]@fit.x)
        e=empirical(np.ones(len(y),bool)); cv=np.array([np.sqrt(np.mean(empirical(groups!=j)[groups==j]**2)) for j in np.unique(groups)])
        print(walls,'unregularized feasible RMSE',np.sqrt(np.mean(e**2)),'rosette aggregate/worst',np.sqrt(np.mean(cv**2)),cv.max())
        if walls!=('periodic','periodic'): continue
        terminal=(fields[-1]@w).reshape(nx,ny,order='F'); np.testing.assert_allclose(out['terminal_field'],terminal,rtol=0,atol=1e-12)
        P=np.linalg.solve(z.T@z+c['nuisance_ridge']*np.eye(z.shape[1]),z.T)
        T=S-z@(P@S); R=y-z@(P@y); residual=T@w-R
        J=lambda u:0.5*(np.linalg.norm(T@u-R)**2+ridge*np.linalg.norm(u)**2)
        jac=lambda u:T.T@(T@u-R)+ridge*u
        H = T.T @ T + ridge * np.eye(len(w))
        ones = np.ones((len(w), 1))
        K = np.block([[H, ones], [ones.T, np.zeros((1, 1))]])
        ref = np.linalg.solve(K, np.r_[T.T @ R, budget])[:-1]
        assert np.all((ref > 0) & (ref < cap)), 'Reference optimum is not interior'
        np.testing.assert_allclose(w, ref, rtol=0, atol=1e-10)
        print('maximum weight error vs independent equality-KKT solve:', np.max(np.abs(w-ref)))
        np.testing.assert_allclose(w,cert['weights'],rtol=0,atol=0)
        nu=cert['nu_sum']; g=jac(w); low=np.array(cert['lam_lower']); high=np.array(cert['lam_upper'])
        dual=J(w)-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
        primal_error=max(abs(w.sum()-budget),max(0,-w.min(),w.max()-cap)); comp=max(abs(low*w).max(),abs(high*(cap-w)).max()); stat=abs(g-nu-low+high).max()
        gap=cert['primal_objective']-cert['dual_lower_bound']; print('primal feasibility, stationarity, complementarity, gap:',primal_error,stat,comp,gap)
        assert min(low.min(),high.min())>=0 and max(primal_error,stat,comp,abs(gap),abs(cert['primal_objective']-J(w)),abs(cert['dual_lower_bound']-dual))<1e-10
    print('independent dense forward solve, objective, and certificate passed')

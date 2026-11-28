import sys, pickle
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import solve_downstream, axis_operator, box_simplex
from _verified_upstream import calibrate_upstream

def dense_axis(k,v,h,periodic):
    n = len(k)
    a = np.zeros((n,n))
    for i in range(n):
        if not periodic and i in (0,n-1):
            a[i,1 if i==0 else n-2] = -k[i]/h**2
        else:
            l,r=(i-1)%n,(i+1)%n
            a[i,l] += -(k[l]+k[i])/(2*h*h)-max(v[i],0)/h
            a[i,r] += -(k[r]+k[i])/(2*h*h)-max(-v[i],0)/h
        a[i,i] = -a[i].sum()
    return a

def audit(h,c,w,nu,cap,budget):
    g=h@w-c
    lo=np.maximum(g-nu,0); hi=np.maximum(nu-g,0)
    errors=np.array([abs(w.sum()-budget), max(0.,-w.min(),w.max()-cap), max(np.max(abs(lo*w)),np.max(abs(hi*(cap-w)))), lo@w+hi@(cap-w)])
    assert errors.max()<1e-8, errors
    return errors

with open('/holdout_task/public_batch.pkl','rb') as f:
    items=pickle.load(f)
maxerrors=np.zeros(4)
for j,item in enumerate(items):
    st=item['study']; cal=calibrate_upstream(dict(st,reference_tracer_pulses=item.get('campaign_diagnostics')))
    nx,ny=st['nx'],st['ny']
    ax,ay=[dense_axis(st['kappa_'+d],st['vel_'+d],st['L'+d]/(st['n'+d]-1),d=='x') for d in 'xy']
    l=np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(st['V_reaction_2d'].ravel(order='F'))
    rhs=np.stack([m.ravel(order='F') for m in st['source_modes']],axis=1)
    ij=st['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]
    replies=[np.linalg.solve(l+mu*np.eye(nx*ny),rhs) for mu in st['mu_shifts']]
    s=np.concatenate([x[idx] for x in replies]); y=cal['y_obs_raw'].ravel(); z=st['Z_nuisance']
    p=np.eye(len(y))-z@np.linalg.solve(z.T@z+st['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    a,b=p@s,p@y; rho=st['source_ridge']; cap,budget=st['u_box'],st['total_budget']
    h=a.T@a+rho*np.eye(a.shape[1]); c=a.T@b
    out=solve_batch(st); w=out['weights']; cert=out['kkt_certificate']; nu=cert['nu_sum']; g=h@w-c
    dn=solve_downstream(cal)
    assert np.array_equal(w,dn['weights'])
    np.testing.assert_array_equal(cert['weights'],w)
    np.testing.assert_allclose(cert['lam_lower'],np.maximum(g-nu,0),rtol=0,atol=1e-10)
    np.testing.assert_allclose(cert['lam_upper'],np.maximum(nu-g,0),rtol=0,atol=1e-10)
    expected_field=(replies[-1]@w).reshape(nx,ny,order='F')
    field_err=np.max(abs(out['terminal_field']-expected_field))
    assert field_err<1e-10, field_err
    obj=(np.linalg.norm(a@w-b)**2+rho*np.linalg.norm(w)**2)/2
    dual=obj-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
    np.testing.assert_allclose([cert['primal_objective'],cert['dual_lower_bound']],[obj,dual],rtol=0,atol=1e-10)
    maxerrors=np.maximum(maxerrors,audit(h,c,w,nu,cap,budget))
    print('public study',j,'field L_inf',field_err,'primal-dual gap',obj-dual)
    for i,pulse in enumerate(cal.get('reference_tracer_pulses') or []):
        yp=pulse['y_pulse_raw'].ravel()
        err=np.linalg.norm(s@pulse['w_pulse']-yp)/np.linalg.norm(yp)
        assert err<3e-5, err
        print('boundary pulse',i,'relative L2',err)

rng=np.random.default_rng(1)
for n in (3,5,15):
    k=rng.uniform(.15,2,n); v=rng.uniform(-2,2,n)
    for periodic in (False,True):
        np.testing.assert_allclose(axis_operator(k,v,.21,periodic).toarray(),dense_axis(k,v,.21,periodic),atol=1e-13)
for j in range(350):
    n=int(rng.integers(3,19)); d=rng.normal(size=(n,n)); h=d.T@d+.04*np.eye(n); c=rng.normal(size=n)*rng.choice([.1,1,10])
    cap=.42; budget=(rng.choice([.00001,.0001,.4,.9999,.99999])*n*cap if j%7 else int(rng.integers(1,n))*cap)
    w,nu=box_simplex(h,c,cap,budget)
    maxerrors=np.maximum(maxerrors,audit(h,c,w,nu,cap,budget))
print('350 budget stress cases plus public: max sum error, box violation, complementarity, gap',maxerrors)

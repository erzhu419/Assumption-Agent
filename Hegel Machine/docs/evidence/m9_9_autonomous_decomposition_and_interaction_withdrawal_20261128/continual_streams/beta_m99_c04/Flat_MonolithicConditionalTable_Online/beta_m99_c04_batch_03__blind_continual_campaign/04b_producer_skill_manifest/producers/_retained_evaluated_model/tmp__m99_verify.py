import importlib.util, itertools, json, pickle
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve
p='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
items=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
log=json.loads(Path('/tmp/hegel_continual_archive/archive_apparatus_log.json').read_text())
print('prefix wall signatures by completed campaign:', [(e['completed_campaign_short_name'],np.mean(e['facility_telemetry_trace'],axis=0)[[2,4]].round(3).tolist()) for e in log])
def axis(k,v,h,rob,kind):
    n=len(k); A=np.zeros((n,n))
    for i in range(n):
        if i in (0,n-1) and kind!='periodic':
            A[i,1 if i==0 else n-2]=-k[i]/h**2
            A[i,i]=k[i]/h**2+(rob/h if kind=='open' else 0.)
        else:
            left,right=(i-1)%n,(i+1)%n
            A[i,left]=-(k[i]+k[left])/(2*h*h)-max(v[i],0)/h
            A[i,right]=-(k[i]+k[right])/(2*h*h)-max(-v[i],0)/h
            A[i,i]=-A[i,left]-A[i,right]
    return A
def reference(s,kind):
    nx,ny=s['nx'],s['ny']; i,j=s['sensor_ij'].T
    ax=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],kind)
    ay=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],kind)
    A=np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
    rhs=np.array(s['source_modes']).transpose(0,2,1).reshape(-1,nx*ny).T
    fields=[solve(A+mu*np.eye(nx*ny),rhs) for mu in s['mu_shifts']]
    return np.concatenate([U[i+nx*j] for U in fields]),fields[-1].T.reshape(-1,ny,nx).transpose(0,2,1)
def project(a,z,ridge):
    return a-z@solve(z.T@z+ridge*np.eye(z.shape[1]),z.T@a,assume_a='pos')
def qp(S,y,s):
    ridge,cap,budget=[s[k] for k in ('source_ridge','u_box','total_budget')]; n=S.shape[1]
    def f(w): return .5*(np.linalg.norm(S@w-y)**2+ridge*(w@w))
    def g(w): return S.T@(S@w-y)+ridge*w
    res=minimize(f,np.full(n,budget/n),jac=g,bounds=[(0,cap)]*n,constraints=[{'type':'eq','fun':lambda w:w.sum()-budget,'jac':lambda w:np.ones(n)}],method='SLSQP',options={'ftol':1e-13,'maxiter':200})
    assert res.success,res.message
    return res.x,f,g
for idx,item in enumerate(items):
    s=item['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; rows=np.arange(len(x)); cv=np.empty_like(y); inv=np.empty_like(x)
    for test in np.array_split(np.random.default_rng(718).permutation(rows),8):
        tr=np.setdiff1d(rows,test); b=m._fit_rosette(x[tr],y[tr]); pred=np.column_stack([m._rosette_basis(x[test],k)@b[k] for k in range(4)]); cv[test]=y[test]-pred; inv[test]=m._invert_rosette(pred,b)
    clean=np.linalg.norm(cv,axis=1)<3
    print(idx,'calibration held-out per-bank max',np.max(abs(cv[clean]),axis=0),'gross rows',rows[~clean],'inverse roundtrip max',np.max(abs(inv-x)))
    assert np.max(abs(cv[clean]))<2e-4 and np.max(abs(inv-x))<1e-9
    yy=m._calibrate(s); b=m._fit_rosette(x,y); xx=yy.reshape(-1,4); rawfit=np.column_stack([m._rosette_basis(xx,k)@b[k] for k in range(4)])
    assert np.max(abs(rawfit.ravel()-s['y_obs_raw']))<1e-10
    out=m.solve_batch(s); w=out['weights']; cert=out['kkt_certificate']; z=s['Z_nuisance']; nr=s['nuisance_ridge']; cap=s['u_box']; bud=s['total_budget']; yp=project(yy,z,nr)
    for kind in ('open','sealed','periodic'):
        D,U=reference(s,kind); S=project(D,z,nr); wr,f,g=qp(S,yp,s)
        scores=[]; ns=len(s['sensor_ij'])
        for h in range(len(s['mu_shifts'])):
            test=np.arange(h*ns,(h+1)*ns); tr=np.setdiff1d(np.arange(len(yy)),test); zz=z[tr]
            wh,_,_=qp(project(D[tr],zz,nr),project(yy[tr],zz,nr),s)
            gamma=solve(zz.T@zz+nr*np.eye(z.shape[1]),zz.T@(yy[tr]-D[tr]@wh),assume_a='pos')
            scores.append(np.sqrt(np.mean((D[test]@wh+z[test]@gamma-yy[test])**2)))
        print(idx,kind,'objective',f(wr),'leave-shift-out RMSE',np.round(scores,7))
        if kind!='open': continue
        Dc,Uc=m._design(s); np.testing.assert_allclose(D,Dc,atol=2e-12); np.testing.assert_allclose(U,Uc,atol=2e-12)
        np.testing.assert_allclose(np.tensordot(w,U,axes=1),out['terminal_field'],atol=2e-12)
        lo,hi=np.array(cert['lam_lower']),np.array(cert['lam_upper']); stationarity=max(abs(g(w)-cert['nu_sum']-lo+hi)); gap=cert['primal_objective']-cert['dual_lower_bound']; comp=max(np.max(abs(w*lo)),np.max(abs((cap-w)*hi)))
        print('cert worst',{'slsqp_weight_difference':float(max(abs(w-wr))),'objective_difference':f(w)-f(wr),'stationarity':stationarity,'gap':gap,'complementarity':comp,'budget_error':abs(w.sum()-bud),'min_w':w.min(),'max_w':w.max()})
        assert abs(w.sum()-bud)<1e-10 and w.min()>-1e-10 and w.max()<cap+1e-10 and lo.min()>=0 and hi.min()>=0
        assert max(stationarity,abs(gap),comp,abs(f(w)-f(wr)),abs(f(w)-cert['primal_objective']))<1e-9
        np.testing.assert_array_equal(w,cert['weights'])

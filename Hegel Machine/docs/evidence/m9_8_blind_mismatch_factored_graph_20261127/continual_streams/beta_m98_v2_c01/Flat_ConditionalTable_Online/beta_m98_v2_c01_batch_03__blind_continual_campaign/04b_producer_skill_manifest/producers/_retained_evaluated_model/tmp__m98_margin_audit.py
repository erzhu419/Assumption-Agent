import importlib.util
import itertools
import pickle
import numpy as np

p='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',p+'batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for item in pickle.load(open(p+'public_batch.pkl','rb')):
    s=item['study']; S,U=m._forward(s); out=m.solve_batch(s); w=out['weights']; k=out['kkt_certificate']
    nx,ny=s['nx'],s['ny']; N=nx*ny; I=np.eye(N); A=np.diag(s['V_reaction_2d'].ravel(order='F'))
    for indices in np.ndindex(nx,ny):
        i,j=indices; row=i+nx*j
        for a,t,n,stride in [('x',i,nx,1),('y',j,ny,nx)]:
            kap=s['kappa_'+a]; v=s['vel_'+a][t]; h=s['L'+a]/(n-1)
            for d in (-1,1):
                u=(t+d)%n; c=(kap[t]+kap[u])/(2*h*h)+max(-d*v,0)/h
                A[row,row]+=c; A[row,row+(u-t)*stride]-=c
    rhs=np.column_stack([x.ravel(order='F') for x in s['source_modes']]); ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]
    F=[np.linalg.solve(A+mu*I,rhs) for mu in s['mu_shifts']]; Sd=np.vstack([f[rows] for f in F]); field=(F[-1]@w).reshape((nx,ny),order='F')
    assert np.allclose(S,Sd,atol=1e-12,rtol=1e-12) and np.allclose(out['terminal_field'],field,atol=1e-12,rtol=1e-12)
    c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; X=np.c_[x,np.ones(len(x))]; ids=np.arange(len(x)); bank=s['row_bank_id']; pulse_metrics={}
    for diagonal in (False,True):
        sweep=[]; pulse=[]
        for test in np.array_split(ids,5):
            tr=np.setdiff1d(ids,test); B=np.linalg.lstsq(X[tr],y[tr],rcond=None)[0]
            if diagonal:
                B=np.zeros_like(B)
                for b in range(4): B[[b,4],b]=np.linalg.lstsq(X[tr][:,[b,4]],y[tr,b],rcond=None)[0]
            sweep.append(np.max(np.sqrt(np.mean((X[test]@B-y[test])**2,axis=0))))
            for q in item['campaign_diagnostics']:
                yc=np.linalg.solve(B[:4].T,(q['y_pulse_raw'].reshape(-1,4)-B[4]).T).T.ravel(); r=Sd@q['w_pulse']-yc
                pulse.extend(np.sqrt(np.mean(r.reshape(len(F),-1)[:,bank[:len(rows)]==b]**2,axis=1)).max() for b in np.unique(bank))
        pulse_metrics[diagonal]=max(pulse); print('diagonal calibration',diagonal,'worst held-out sweep bank RMSE',max(sweep),'worst pulse shift-bank RMSE',max(pulse))
    assert pulse_metrics[False]<2e-5
    Z=s['Z_nuisance']; P=np.eye(len(Z))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T); D=P@Sd; yy=P@m._calibrate(s); lam=s['source_ridge']; cap=s['u_box']; budget=s['total_budget']; H=D.T@D+lam*np.eye(len(w)); b=D.T@yy
    def J(v):
        r=D@v-yy; return float((r@r+lam*(v@v))/2)
    best=np.inf
    for state in itertools.product((-1,0,1),repeat=len(w)):
        st=np.asarray(state); free=np.flatnonzero(st==0); v=cap*(st==1).astype(float); rem=budget-v.sum()
        if free.size and -1e-12<=rem<=cap*free.size+1e-12:
            ones=np.ones((free.size,1)); K=np.block([[H[np.ix_(free,free)],ones],[ones.T,np.zeros((1,1))]])
            v[free]=np.linalg.solve(K,np.r_[b[free]-(H@v)[free],rem])[:-1]
        elif free.size or abs(rem)>1e-12: continue
        if np.min(v)>=-1e-12 and np.max(v)<=cap+1e-12: best=min(best,J(v))
    g=H@w-b; lo=np.array(k['lam_lower']); hi=np.array(k['lam_upper']); nu=k['nu_sum']; gap=J(w)-k['dual_lower_bound']; stat=np.max(abs(g-nu-lo+hi)); comp=max(np.max(abs(lo*w)),np.max(abs(hi*(cap-w))))
    print('dense design max difference',np.max(abs(S-Sd)),'field max difference',np.max(abs(out['terminal_field']-field)),'objective minus enumerated optimum',J(w)-best,'certificate gap, stationarity, complementarity',gap,stat,comp)
    assert abs(J(w)-best)<1e-10 and abs(gap)<1e-10 and max(stat,comp,abs(w.sum()-budget))<1e-10
    assert np.allclose(k['weights'],w,atol=0,rtol=0) and abs(k['primal_objective']-J(w))<1e-10 and w.min()>=-1e-12 and w.max()<=cap+1e-12

import importlib.util,pickle
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; o=m.solve_batch(s); w=o['weights']; q=o['kkt_certificate']; nx,ny=s['nx'],s['ny']
def dense_axis(a,n):
    k,v=s['kappa_'+a],s['vel_'+a]; h=s['L'+a]/(n-1); r=s['robin_'+a]
    A=np.zeros((n,n))
    for i,j in [(0,1),(n-1,n-2)]:
        A[i,i]=k[i]/h**2+r/h; A[i,j]=-k[i]/h**2
    for i in range(1,n-1):
        km,kp=(k[i]+k[i-1])/2,(k[i]+k[i+1])/2
        A[i,i-1]=-km/h**2-max(v[i],0)/h; A[i,i+1]=-kp/h**2-max(-v[i],0)/h
        A[i,i]=(kp+km)/h**2+abs(v[i])/h
    return A
L=np.kron(np.eye(ny),dense_axis('x',nx))+np.kron(dense_axis('y',ny),np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
rhs=np.column_stack([u.ravel(order='F') for u in s['source_modes']]); ij=s['sensor_ij']; slots=ij[:,0]+nx*ij[:,1]
blocks=[]
for shift in s['mu_shifts']:
    F=np.linalg.solve(L+shift*np.eye(nx*ny),rhs); blocks.append(F[slots])
expected=(F@w).reshape((nx,ny),order='F'); terr=np.max(abs(expected-o['terminal_field'])); assert terr<1e-11
A=np.vstack(blocks); y=m._calibrate(s); z=s['Z_nuisance']; profile=np.eye(len(z))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
assert np.max(abs(A-m._transport(s)[0]))<1e-11
A,r=profile@A,profile@y
res=A@w-r; rho=s['source_ridge']; g=A.T@res+rho*w; cap=s['u_box']; budget=s['total_budget']; nu=q['nu_sum']; lo,hi=np.array(q['lam_lower']),np.array(q['lam_upper']); J=(res@res+rho*(w@w))/2
D=J-g@w+nu*budget+cap*np.minimum(g-nu,0).sum()
stat=np.max(abs(g-nu-lo+hi)); comp=max(np.max(abs(w*lo)),np.max(abs((cap-w)*hi)))
assert np.array_equal(q['weights'],w) and abs(w.sum()-budget)<1e-11 and min(w)>=0 and max(w)<=cap
assert min(lo)>=0 and min(hi)>=0 and max(abs(J-q['primal_objective']),abs(D-q['dual_lower_bound']),stat,comp,abs(J-D))<1e-11
print('terminal max error',terr,'stationarity',stat,'complementarity',comp,'gap',J-D)
rng=np.random.default_rng(172); worst=0.; hits=[0,0]
for n in [2,8,20]:
    for frac in [0,0.015,0.25,0.5,0.985,1]:
        cap=.42; budget=n*cap*frac
        for rep in range(25):
            B=rng.normal(size=(n,n)); H=B.T@B+.04*np.eye(n); c=rng.normal(size=n)*4
            w,nu=m._box_simplex(H,c,cap,budget); g=H@w+c
            hits[0]+=np.count_nonzero(w==0); hits[1]+=np.count_nonzero(w==cap)
            err=max(abs(w.sum()-budget),max(0.,-w.min(),w.max()-cap),abs(g@w-nu*budget-cap*np.minimum(g-nu,0).sum()))
            worst=max(worst,err)
assert worst<1e-9 and min(hits)>0
print('450 box-simplex stress cases: worst feasibility/certificate error',worst,'lower/upper bound hits',hits)

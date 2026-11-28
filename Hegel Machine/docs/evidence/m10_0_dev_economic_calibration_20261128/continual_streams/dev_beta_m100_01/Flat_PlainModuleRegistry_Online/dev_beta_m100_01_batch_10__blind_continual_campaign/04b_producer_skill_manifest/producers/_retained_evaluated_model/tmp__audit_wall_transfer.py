import itertools, pickle, sys
import numpy as np
from scipy import sparse
from scipy.linalg import null_space
from scipy.sparse.linalg import splu
sys.path.insert(0, '/holdout_task')
import batch_solver, downstream_solver
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f:
    study = pickle.load(f)[0]['study']
s = calibrate_upstream(study)
y, Z = s['y_obs_raw'], s['Z_nuisance']
nx,ny = s['nx'],s['ny']; cap,B=s['u_box'],s['total_budget']; m=len(s['source_modes']); ix,iy=s['sensor_ij'].T
rhs=np.column_stack([a.ravel(order='F') for a in s['source_modes']]); nobs=len(ix)
I=sparse.eye(nx*ny,format='csc'); reaction=sparse.diags(s['V_reaction_2d'].ravel(order='F'))
def axis(k,v,h,r,ends):
    n=len(k); A=np.zeros((n,n))
    for i in range(n):
        t=ends[0] if i==0 else ends[1] if i==n-1 else 'p'
        if t=='p':
            for j, adv in (((i-1)%n,max(v[i],0)),((i+1)%n,max(-v[i],0))):
                A[i,j]-=(k[i]+k[j])/(2*h*h)+adv/h
        else: A[i,1 if i==0 else n-2]=-k[i]/(h*h)
        A[i,i]=-A[i].sum()+(r/h if t=='r' else 0)
    return sparse.csc_matrix(A)
axes={d:{t:axis(s['kappa_'+d],s['vel_'+d],s['L'+d]/(s['n'+d]-1),s['robin_'+d],t) for t in map(''.join,itertools.product('prs',repeat=2))} for d in 'xy'}
models={}
for tx,ty in itertools.product(axes['x'],axes['y']):
    L=sparse.kron(sparse.eye(ny),axes['x'][tx],format='csc')+sparse.kron(axes['y'][ty],sparse.eye(nx),format='csc')+reaction
    models[tx+ty]=np.vstack([splu(L+shift*I).solve(rhs)[ix+nx*iy] for shift in s['mu_shifts']])
assert np.max(np.abs(models['ppss']-downstream_solver._forward(s)[0])) < 1e-12
N=null_space(np.ones((1,m))); w0=np.full(m,B/m)
def residual(S, train):
    A,z,yy=S[train],Z[train],y[train]
    coef=np.linalg.lstsq(np.column_stack((A@N,z)), yy-A@w0,rcond=None)[0]
    w=w0+N@coef[:m-1]
    if w.min()<-1e-10 or w.max()>cap+1e-10:
        data=np.column_stack((A,yy)); p=data-z@np.linalg.lstsq(z,data,rcond=None)[0]
        w,_=downstream_solver._qp(p[:,:m],p[:,m],1e-10,cap,B)
    gamma=np.linalg.lstsq(z,yy-A@w,rcond=None)[0]
    return y-S@w-Z@gamma
rows=np.arange(len(y)); family=(rows//4)%(nobs//4); shift=rows//nobs
for name, outer, inner in [('rosette',family,family),('shift',shift,family)]:
    selected=[]; scores=[]; rivals=[]
    for group in np.unique(outer):
        train=outer!=group; test=~train
        def score(S):
            ss=0.
            for j in np.unique(inner[train]):
                a=train&(inner!=j); b=train&(inner==j)
                ss+=np.sum(residual(S,a)[b]**2)
            return ss
        winner=min(models,key=lambda k:score(models[k])); selected.append(winner)
        scores.append(np.sqrt(np.mean(residual(models[winner],train)[test]**2)))
        rivals.append(np.sqrt(np.mean(residual(models['pppp'],train)[test]**2)))
    print(name,'inner-CV winners',selected,'outer worst nested vs archived',max(scores),max(rivals))
for j in np.unique(s['row_bank_id']):
    test=s['row_bank_id']==j
    print('left-out nuisance bank',j,'RMSE mixed vs archived',*[np.sqrt(np.mean(residual(models[k],~test)[test]**2)) for k in ('ppss','pppp')])
w=batch_solver.solve_batch(study); c=w['kkt_certificate']; print('isolated public solve: mass and primal-dual gap',sum(w['weights']),c['primal_objective']-c['dual_lower_bound'])
assert abs(sum(w['weights'])-B)<1e-12 and abs(c['primal_objective']-c['dual_lower_bound'])<1e-10

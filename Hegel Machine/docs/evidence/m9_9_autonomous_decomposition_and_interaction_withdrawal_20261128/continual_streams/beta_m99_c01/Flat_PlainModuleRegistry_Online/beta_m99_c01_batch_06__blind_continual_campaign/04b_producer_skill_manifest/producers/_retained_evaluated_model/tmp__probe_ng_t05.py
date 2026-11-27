exec(open('/tmp/compare_walls.py').read().split('for a,b in itertools.product')[0])
from scipy.optimize import minimize
bank=np.asarray(s['row_bank_id']); shift=np.repeat(np.arange(len(mus)),len(x)); edge=np.tile((x==0)|(x==nx-1)|(yidx==0)|(yidx==ny-1),len(mus))

def wall(d, t):
    # Interpolate whole boundary rows; interior coefficients remain identical.
    return (1-t)*axis(d,'R')+t*axis(d,'P')

def design(a,b):
    L=kron(eye(ny),a)+kron(b,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F')); I=eye(nx*ny)
    return np.vstack([splu((L+mu*I).tocsc()).solve(rhs)[x+nx*yidx] for mu in mus])

def scores(A, penalty):
    ans=[]
    for groups in (shift,bank):
        err=np.zeros(len(y))
        for i in np.unique(groups):
            te=groups==i; tr=~te; Z=z[tr]; D=np.c_[A[tr],y[tr]]
            P=np.linalg.solve(Z.T@Z+penalty*s['nuisance_ridge']*eyez,Z.T@D); T=D-Z@P; X,t=T[:,:-1],T[:,-1]; r=penalty*ridge
            def fg(w):
                e=X@w-t; return .5*(e@e+r*(w@w)),X.T@e+r*w
            sol=minimize(fg,np.full(len(modes),budget/len(modes)),jac=True,method='SLSQP',bounds=[(0,cap)]*len(modes),constraints={'type':'eq','fun':lambda w:w.sum()-budget,'jac':lambda w:np.ones_like(w)},options={'ftol':1e-13}); assert sol.success,sol.message
            err[te]=A[te]@sol.x+z[te]@(P[:,-1]-P[:,:-1]@sol.x)-y[te]
        ans.extend([max(np.sqrt(np.mean(err[groups==g]**2)) for g in np.unique(groups)),np.sqrt(np.mean(err[edge]**2))])
    return ans

print('ng_t05 | columns: worst shift, boundary with shifts held out, worst bank, boundary with banks held out')
results={}
for name,a,b in [('Robin baseline',axis('x','O'),axis('y','O')),('periodic x, reflecting y',wall('x',1),wall('y',0)),('periodic x, half-strength periodic y',wall('x',1),wall('y',.5)),('periodic x and y',wall('x',1),wall('y',1)),('half-strength periodic x, periodic y',wall('x',.5),wall('y',1))]:
    A=design(a,b); results[name]=np.array([scores(A,p) for p in (1,0)]); print(name, 'specified penalties:',results[name][0].round(7),'no penalties:',results[name][1].round(7))
assert np.all(results['periodic x and y']<results['Robin baseline']), 'The periodic revision does not resolve the Robin mismatch in all audited slices.'

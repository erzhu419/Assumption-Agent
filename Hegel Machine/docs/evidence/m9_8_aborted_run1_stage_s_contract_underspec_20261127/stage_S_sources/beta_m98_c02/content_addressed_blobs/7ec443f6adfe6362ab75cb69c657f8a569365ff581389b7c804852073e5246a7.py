"""Calibrated chamber inversion and box-simplex optimality certificate."""
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


def _features(x, level):
    cols = [np.ones(len(x)), *x.T]
    if level >= 1:
        cols.extend((x*x).T)
        cols.extend((x**3).T)
    if level >= 2:
        cols.extend(x[:,i]*x[:,j] for i in range(4) for j in range(i+1,4))
    if level >= 3:
        cols.extend(x[:,i]**2*x[:,j] for i in range(4) for j in range(4) if i != j)
        cols.extend(x[:,i]*x[:,j]*x[:,k] for i in range(4) for j in range(i+1,4) for k in range(j+1,4))
    return np.column_stack(cols)


def _calibration(sweeps):
    x, y = [np.asarray(sweeps[k],float) for k in ('ref_rosette','raw_rosette')]
    # Use the paired rosettes to estimate crosstalk. Stratify the validation
    # folds so that every fit sees single-channel as well as mixed inputs.
    order = np.argsort(np.count_nonzero(x,axis=1),kind='stable')
    folds = np.empty(len(x),int)
    folds[order] = np.arange(len(x)) % 5
    designs = [_features(x,k) for k in range(4)]
    scores = []
    for f in designs:
        scores.append(max(np.sqrt(np.mean((f[folds==j] @ np.linalg.lstsq(f[folds!=j],(y-x)[folds!=j],rcond=None)[0] - (y-x)[folds==j])**2)) for j in range(5)))
    level = next(i for i,v in enumerate(scores) if v <= min(scores)*1.12)
    coef = np.linalg.lstsq(designs[level],y-x,rcond=None)[0]
    affine = (np.eye(4)+coef[1:5]).T

    def invert(raw):
        y = np.asarray(raw).reshape(-1,4)
        x = np.linalg.solve(affine,(y-coef[0]).T).T
        if level:
            for _ in range(12):
                r = x + _features(x,level)@coef - y
                if np.max(np.abs(r)) < 1e-12:
                    break
                jac = np.broadcast_to(affine,(len(x),4,4)).copy()
                for i in range(4):
                    d = np.zeros(4); d[i] = 1e-5
                    jac[:,:,i] = np.eye(4)[:,i] + ((_features(x+d,level)-_features(x-d,level))@coef)/(2e-5)
                x -= np.linalg.solve(jac,r[:,:,None])[:,:,0]
        return x.ravel()
    return invert


def _axis(k,v,h):
    k,v = np.asarray(k),np.asarray(v)
    d = (k[:-1]+k[1:])/(2*h*h)
    lower = -d-np.maximum(v[1:],0)/h
    upper = -d-np.maximum(-v[:-1],0)/h
    return diags([lower,np.r_[-upper,0]+np.r_[0,-lower],upper],[-1,0,1],format='csc')


class _Chamber:
    def __init__(self,s):
        nx,ny = self.shape = s['nx'],s['ny']
        hx,hy = s['Lx']/(nx-1),s['Ly']/(ny-1)
        ax,ay = [_axis(s['kappa_'+k],s['vel_'+k],h) for k,h in zip('xy',(hx,hy))]
        self.A = csc_matrix(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(np.asarray(s['V_reaction_2d']).ravel(order='F')))
        self.I = eye(nx*ny,format='csc')
        x,y = np.indices(self.shape)
        self.edges = np.column_stack([a.ravel(order='F') for a in (x==0,x==nx-1,y==0,y==ny-1)]).astype(float)
        self.slopes = self.edges * np.column_stack([a.ravel(order='F') for a in (2*y/(ny-1)-1,2*y/(ny-1)-1,2*x/(nx-1)-1,2*x/(nx-1)-1)])
        self.nominal = np.repeat([s['robin_x']/hx,s['robin_y']/hy],2)
        ij = np.asarray(s['sensor_ij']); self.sensors = ij[:,0]+nx*ij[:,1]
        self.shifts = np.asarray(s['mu_shifts'])
        self.rhs = np.asarray(s['source_modes']).reshape(-1,nx*ny,order='F').T.copy()

    def response(self,delta,rhs):
        A = self.A + diags(delta,format='csc')
        observations = []
        for shift in self.shifts:
            field = splu(A+shift*self.I).solve(rhs)
            observations.append(field[self.sensors])
        return np.concatenate(observations),field


def _fit_boundary(c,s,y,cal):
    E = c.edges
    pulses = s.get('reference_tracer_pulses')
    if pulses:
        # Diagnostic mixtures provide a direct boundary measurement, with
        # independent walls and the spatial mode indicated by pulse holdouts.
        F = np.column_stack((E,c.slopes[:,0]))
        W = np.column_stack([p['w_pulse'] for p in pulses])
        R = c.rhs@W
        Y = np.column_stack([cal(p['y_pulse_raw']) for p in pulses])
        b = least_squares(lambda b:(c.response(F@b,R)[0]-Y).ravel(),np.r_[c.nominal,0.],gtol=1e-11).x
        return F@b

    # Profile the budget equality and nuisance before estimating leakage.
    # Noisy observations cannot support the same correction as clean data.
    F = np.column_stack((E,c.slopes))
    R = c.rhs
    Z = np.asarray(s['Z_nuisance'])
    m = R.shape[1]
    w0 = np.full(m,s['total_budget']/m)
    Q = np.eye(m)[:,:-1]-np.eye(m)[:,[-1]]
    nominal = E@c.nominal

    def profile(delta):
        S = c.response(delta,R)[0]
        D = np.column_stack((S@Q,Z))
        r = y-S@w0
        return r-D@np.linalg.lstsq(D,r,rcond=None)[0]

    scale = max(3e-5,2.5*np.sqrt(np.mean(profile(nominal)**2)))
    penalties = scale*np.repeat([1.,2.5],4)
    limit = np.maximum(abs(c.nominal),1.)
    span = np.r_[limit,limit*0.5]
    opt = least_squares(lambda p:np.r_[profile(nominal+F@p),penalties*p],
                        np.zeros(8),bounds=(-0.45*span,0.6*span),gtol=1e-11)
    return nominal+F@opt.x


def _quadratic(A,r,ridge,budget,cap):
    m = A.shape[1]
    H = A.T@A+ridge*np.eye(m)
    g = -A.T@r
    cap = np.broadcast_to(np.asarray(cap,float),(m,))
    objective = lambda w: 0.5*np.linalg.norm(A@w-r)**2+0.5*ridge*(w@w)
    def certificate(w,nu):
        q = H@w+g+nu
        lo = np.where(w==0,np.maximum(q,0),0)
        hi = np.where(w==cap,np.maximum(-q,0),0)
        d = q-lo+hi
        p = objective(w)
        lower = p + nu*(w.sum()-budget)-lo@w+hi@(w-cap)-0.5*d@np.linalg.solve(H,d)
        return dict(weights=w,nu_sum=float(nu),lam_lower=lo,lam_upper=hi,primal_objective=float(p),dual_lower_bound=float(lower))

    if budget == 0 or budget == cap.sum():
        w = np.zeros(m) if budget == 0 else cap.copy()
        grad = H@w+g
        return certificate(w,-grad.min() if budget==0 else -grad.max())
    opt = minimize(lambda w:0.5*w@H@w+g@w,cap*(budget/cap.sum()),jac=lambda w:H@w+g,
                   bounds=list(zip(np.zeros(m),cap)),constraints={'type':'eq','fun':lambda w:w.sum()-budget,'jac':lambda w:np.ones(m)},
                   method='SLSQP',options={'ftol':1e-13,'maxiter':300})
    # Snap approximate active bounds and solve the equality-constrained
    # KKT system on the free coordinates; release an incorrect bound.
    side = np.where(opt.x < 1e-8,-1,np.where(cap-opt.x < 1e-8,1,0))
    for _ in range(4*m+4):
        free = side==0
        w = np.where(side==1,cap,0.)
        if not free.any():
            rest = budget-w.sum(); grad = H@w+g
            down = np.flatnonzero(side==1); up = np.flatnonzero(side==-1)
            i = up[np.argmin(grad[up])] if len(up) else -1
            j = down[np.argmax(grad[down])] if len(down) else -1
            if abs(rest)<1e-12 and i>=0 and j>=0 and grad[i]>=grad[j]-1e-12:
                return certificate(w,-0.5*(grad[i]+grad[j]))
            if rest>=-1e-12 and i>=0: side[i]=0
            if rest<=1e-12 and j>=0: side[j]=0
            continue
        hf = H[np.ix_(free,free)]
        a,b = np.linalg.solve(hf,np.column_stack((-(H@w+g)[free],np.ones(free.sum())))).T
        nu = (a.sum()+w.sum()-budget)/b.sum()
        w[free] = a-nu*b
        invalid = np.where(free,np.maximum(-w,w-cap),-np.inf)
        i = invalid.argmax()
        if invalid[i]>1e-12:
            side[i] = -1 if w[i]<0 else 1
            continue
        q = H@w+g+nu
        violation = np.where(side!=0,side*q,-np.inf)
        i = violation.argmax()
        if violation[i] > 1e-11:
            side[i]=0
            continue
        return certificate(np.clip(w,0,cap),nu)
    raise RuntimeError('Box-simplex KKT refinement did not converge')


def solve_batch(study: dict) -> dict:
    s = study
    c = _Chamber(s)
    calibrate = _calibration(s['sensor_calibration_sweeps'])
    y = calibrate(s['y_obs_raw'])
    diag = _fit_boundary(c,s,y,calibrate)
    S,U = c.response(diag,c.rhs)
    Z = np.asarray(s['Z_nuisance'])
    Pz = np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
    A,r = S-Z@(Pz@S),y-Z@(Pz@y)
    cert = _quadratic(A,r,s['source_ridge'],s['total_budget'],s['u_box'])
    w = cert['weights']
    return dict(weights=w,terminal_field=(U@w).reshape(c.shape,order='F'),kkt_certificate=cert)

import numpy as np
from scipy.optimize import least_squares

# Calibration coefficients estimated from the public reference panels.
CAL = np.array([[0.033827479672666985, 0.07502994949014716, 0.03556305388690401, -0.10920335674376619], [0.9809307101862607, 0.8629990455768406, 1.1573998900759128, 1.2002882102717807], [-0.31999471283555536, -0.31999428914834527, -0.3199574603474531, -0.3199905103871267], [0.31999582315928554, 0.3199835435188928, 0.3200148733329416, 0.3200118232407754], [0.1600124998192269, 0.15996380454409756, 0.15999319730988887, 0.1599900206487814]])
Q1 = np.array([0.8444230770112555, -0.15774263453922568])
C1 = np.array([[0.08326011432424367, 0.06829789349557093, 0.04102278316560223, 0.030190593886713187], [-0.09090505019568829, -0.3794373611692894, -2.1183688571394956, 0.7529232648949882], [2.103052405384543, -0.6120794597854645, -0.19762994861113498, -0.6077585328589148], [0.696333714798857, 2.1244236743953153, -0.28223555699490804, 0.35714874324838525], [0.5392346967343538, -0.4152752361446725, 0.776606145736224, 2.035328363589207]])
Q2 = np.array([0.3048459992624531])
C2 = np.array([[-0.1207648052269579, 0.1360333597485881, 0.11783672982901112, -0.029565077901737787], [-0.20304870862171004, -0.02471274076998639, -0.2898835380063854, -0.32775246826751175], [-0.11059919195815693, -0.37286054754508563, 0.255458694723335, -0.1294096507969399], [-0.01617743152819192, -0.28148548428968023, -0.2777258505751167, 0.27689821932616077], [-0.42377776906819997, 0.11986388713245974, 0.08283979766434472, 0.1801610153684999], [-0.037660355995836454, -0.005043739307974342, -0.05307504446828029, -0.06270179361098874], [-0.020558102803073297, -0.06962559548665995, 0.0472061949417082, -0.02325426754559209], [-0.0026717784963122397, -0.053143690113138374, -0.05081724531673, 0.05031377363364097], [-0.07841490412041142, 0.022668912580626355, 0.015059750981176031, 0.033010160891762085], [-0.07420465741751027, -0.008180795646081856, -0.10688233259644535, -0.11723769188038459], [-0.0402948846823113, -0.135793892319241, 0.09366734585008932, -0.04838078767792691], [-0.006409742927532261, -0.10144049928549337, -0.10234195017860293, 0.10256147157866055], [-0.1551631385490583, 0.04307952266761627, 0.030664858413288018, 0.06634480135008766]])
OFFSET = np.array([0.06766971726814551, 0.007561306101974368, 0.08093317019163747, -0.08329542359914982])
SCALE = np.array([1.1500970782315993, 1.1491194763935682, 1.1602102016682996, 1.305159236516379])


def _one(z):
    return np.column_stack((np.ones(len(z)), z))


def _curvature(z):
    h = z*z
    return np.column_stack((_one(z), h, h/(1+Q2[0]*h))) @ C2


def _decode(w, n):
    c, d = w[:, :4], w[:, 4:]
    inv = np.linalg.inv(C1[1:])
    def latent(p):
        a = (c*np.exp(p[:4])+p[4:]-C1[0])@inv
        # Invert the softsign map on its central monotone branch.
        b = abs(a)
        q, r = Q1
        B = 1+r-q*b
        disc = B*B+4*r*q*b
        return np.sign(a)*2*b/np.maximum(B+np.sqrt(np.maximum(disc, 1e-12)), .02), disc
    def residual(p):
        z, disc = latent(p)
        v = _curvature(z)
        v = (v-v[:n].mean(0))/np.maximum(v[:n].std(0), .1)
        return np.r_[(v-d).ravel(), 8*np.minimum(disc, 0).ravel()]
    low = np.r_[np.full(4,-3.4),C1[0]-.65]
    high = np.r_[np.full(4,.2),C1[0]+.65]
    best = None
    for scale in (.22, .12, .32, .45):
        p0 = np.r_[np.full(4,np.log(scale)), C1[0]]
        opt = least_squares(residual, p0, bounds=(low,high), max_nfev=110)
        if best is None or opt.cost < best.cost:
            best = opt
        if opt.cost < 1e-13:
            break
    # Resolve any rare folds of the softsign using both panel maps together.
    z, _ = latent(best.x)
    def full(p):
        zz = p[8:].reshape(-1,4)
        soft = _one(zz/(1+Q1[0]*abs(zz))+Q1[1]*zz)@C1
        v = _curvature(zz)
        v = (v-v[:n].mean(0))/np.maximum(v[:n].std(0), .1)
        return np.r_[(4*(soft-c*np.exp(p[:4])-p[4:8])).ravel(), (v-d).ravel()]
    if best.cost > 1e-12:
        opt = least_squares(full, np.r_[best.x,z.ravel()], max_nfev=45, ftol=1e-10, gtol=1e-10)
        z = opt.x[8:].reshape(-1,4)
        loss = np.sqrt(np.mean(opt.fun**2))
    else:
        loss = np.sqrt(np.mean(best.fun**2))
    return z, loss


def _calibrated(z):
    x = z*SCALE+OFFSET
    z = z.copy()
    j = np.roll(np.arange(4),-1)
    for _ in range(15):
        v = z[:,j]
        r = CAL[0]+CAL[1]*z+CAL[2]*v+CAL[3]*z*v+CAL[4]*z**3-x
        jac = np.zeros((len(z), 4, 4))
        k = np.arange(4)
        jac[:,k,k] = CAL[1]+CAL[3]*v+3*CAL[4]*z*z
        jac[:,k,j] = CAL[2]+CAL[3]*z
        z -= np.clip(np.linalg.solve(jac,r[...,None])[...,0],-.65,.65)
    soft = _one(z/(1+Q1[0]*abs(z))+Q1[1]*z) @ C1
    return np.column_stack((soft,_curvature(z)))


def run_node(inputs: dict) -> dict:
    w = np.asarray(inputs['node_00']['features'], dtype=float)
    t = inputs['node_01']
    targets, basis = np.asarray(t['targets']), np.asarray(t['basis'])
    baseline = np.asarray(inputs['node_02']['fit'])
    n = len(targets)
    z, loss = _decode(w, n)
    pred = baseline.copy()
    if loss < .025:
        f = _calibrated(z)
        # Quadratic response in the recovered panel coordinates.
        f = (f-f[:n].mean(0))/np.maximum(f[:n].std(0), .1)
        i, j = np.triu_indices(f.shape[1])
        F = np.column_stack((_one(f), f[:,i]*f[:,j]))
        U, D, V = np.linalg.svd(F[:n], full_matrices=False)
        ridge = .02
        model = ((F[n:] @ V.T)*(D/(D*D+ridge))) @ (U.T @ targets)
        pred = .998*model+.002*baseline
    y = pred[:,0]
    state = (pred[:,1:] @ np.linalg.pinv(basis).T) @ basis.T
    diag = np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diag}

"""Invert a borehole voltage survey with separable contact and anomaly terms."""

import numpy as np
from scipy.linalg import helmert
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize


class _Profile:
    """Least-squares profile with nonnegative anomaly amplitude."""

    def __init__(self, design, voltage, keep, tx, rx, length, depth, radius):
        self.design, self.voltage = design[keep], voltage[keep]
        self.q = np.linalg.qr(self.design, mode='reduced')[0]
        self.y = self.project(self.voltage)
        self.tx, self.rx = tx[keep], rx[keep]
        self.length, self.depth = length, depth
        self.radius2 = radius**2
        self.scale = max(self.y @ self.y, 1e-20)

    def project(self, value):
        return value - np.einsum('ij,j->i', self.q, np.einsum('ij,i->j', self.q, value))

    def kernel(self, x, z):
        t = z - self.tx
        r = z - self.rx
        a = x*x + t*t + self.radius2
        b = (self.length-x)**2 + r*r + self.radius2
        return 1.0 / np.sqrt(a*b), a, b, t, r

    def objective(self, u):
        x, z = u * [self.length, self.depth]
        k, a, b, t, r = self.kernel(x, z)
        p = self.project(k)
        amp = max(0., p @ self.y) / max(p @ p, 1e-30)
        e = self.y - amp*p
        dx = k*((self.length-x)/b - x/a)
        dz = -k*(t/a + r/b)
        g = (-2*amp/self.scale) * np.array([self.length*(e @ dx), self.depth*(e @ dz)])
        return (e @ e)/self.scale, g

    def search(self, radius, previous=None, posterior=False):
        nx = int(np.clip(np.ceil(2*self.length/radius)+1, 19, 65))
        nz = int(np.clip(np.ceil(2*self.depth/radius)+1, 25, 97))
        gx, gz = np.linspace(0., 1., nx), np.linspace(0., 1., nz)
        xx, zz = np.meshgrid(gx, gz, indexing='ij')
        uu = np.column_stack((xx.ravel(), zz.ravel()))
        x = self.length*uu[:, 0, None]
        z = self.depth*uu[:, 1, None]
        k = self.kernel(x, z)[0]
        # Compute the projected norms without forming a second full grid matrix.
        qk = np.einsum('ij,jk->ik', k, self.q)
        ky = np.einsum('ij,j->i', k, self.y)
        norm2 = np.einsum('ij,ij->i', k, k) - np.einsum('ij,ij->i', qk, qk)
        score = np.maximum(ky, 0.)**2 / np.maximum(norm2, 1e-30)
        if posterior:
            # Resolve the narrow depth likelihood independently of grid spacing.
            x0, z0 = previous * [self.length, self.depth]
            k, a, b, t, r = self.kernel(x0,z0)
            pk = self.project(k)
            amp = max(0.,pk @ self.y)/max(pk @ pk,1e-30)
            J = np.column_stack((pk, self.project(amp*k*((self.length-x0)/b-x0/a)),
                                 self.project(-amp*k*(t/a+r/b))))
            sse0 = self.objective(previous)[0]*self.scale
            dof = max(1,self.y.size-self.q.shape[1]-3)
            cov = np.linalg.pinv(J.T@J,rcond=1e-13)*sse0/dof
            if not (0.05 < cov[1,1]**.5 < 0.50*self.length):
                return previous
            # Adapt the depth band to the measured joint uncertainty.
            lo, hi = max(0.,z0-9*cov[2,2]**.5), min(self.depth,z0+9*cov[2,2]**.5)
            if hi-lo < 1.e-6:
                return previous
            xx=np.linspace(0.,self.length,181)
            zz=np.linspace(lo,hi,145)
            # Expand the depth band around the linearized likelihood ridge.
            zz=np.clip(zz[None,:]+np.clip(cov[1,2]/cov[1,1]*(xx[:,None]-x0),-(hi-lo)/2,(hi-lo)/2),0.,self.depth)
            weight=np.empty_like(zz)
            for start in range(0,len(xx),12):
                block = slice(start,start+12)
                zs = zz[block]
                x = np.broadcast_to(xx[block,None],zs.shape).ravel()[:,None]
                k = self.kernel(x,zs.ravel()[:,None])[0]
                qk=np.einsum('ij,jk->ik',k,self.q)
                ky=np.einsum('ij,j->i',k,self.y)
                norm2=np.einsum('ij,ij->i',k,k)-np.einsum('ij,ij->i',qk,qk)
                sse=np.maximum(self.scale-np.maximum(ky,0.)**2/np.maximum(norm2,1e-30),1e-30)
                weight[block]=np.exp(np.clip(-.5*dof*np.log(sse/max(sse0,1e-30)),-745.,300.)).reshape(zs.shape)
            slice_w=np.trapz(weight,zz,axis=1)
            norm=np.trapz(slice_w,xx)
            if norm == 0.:
                return previous
            # Pick the horizontal interval with greatest supported mass at
            # the requested spatial resolution. Average depth within it.
            dx = xx[1]-xx[0]
            half = min(2.,self.length/2)
            xx_ext = np.r_[xx[0],xx[1:]]
            cumulative = np.r_[0.,np.cumsum(.5*dx*(slice_w[1:]+slice_w[:-1]))]
            mass = np.interp(xx+half,xx_ext,cumulative)-np.interp(xx-half,xx_ext,cumulative)
            j = np.argmax(mass)
            cx = xx[j]
            if 0 < j < len(xx)-1:
                left, center, right = mass[j-1:j+2]
                curvature = left-2*center+right
                if curvature < 0:
                    cx += np.clip(.5*(left-right)/curvature,-.5,.5)*dx
            sz = np.trapz(weight*zz,zz,axis=1)
            cumulative_z = np.r_[0.,np.cumsum(.5*dx*(sz[1:]+sz[:-1]))]
            mass_center = np.interp(cx+half,xx,cumulative)-np.interp(cx-half,xx,cumulative)
            cz = (np.interp(cx+half,xx,cumulative_z)-np.interp(cx-half,xx,cumulative_z))/mass_center
            return np.array([cx/self.length,cz/self.depth])
        surface = -score.reshape(nx, nz)
        local = np.flatnonzero((surface == minimum_filter(surface, size=3)).ravel())
        local = local[np.argsort(surface.ravel()[local])[:10]]
        starts = list(uu[local])
        if previous is not None:
            starts.append(previous)
        best = None
        for start in starts:
            fit = minimize(self.objective, start, jac=True, method='L-BFGS-B',
                           bounds=[(0., 1.), (0., 1.)],
                           options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 100})
            if best is None or fit.fun < best.fun:
                best = fit
        return best.x


def invert_ert_anomaly(survey: dict) -> dict:
    """Return the anomaly coordinates and zero-sum receiver contact voltages."""
    length, depth = float(survey['L_m']), float(survey['D_m'])
    radius = float(survey.get('a0_m', 6.))
    tx = np.asarray(survey['z_tx_m'], dtype=float)
    rx = np.asarray(survey['z_rx_m'], dtype=float)
    voltage = np.asarray(survey['V_obs'], dtype=float).ravel()
    ti = np.repeat(np.arange(tx.size), rx.size)
    ri = np.tile(np.arange(rx.size), tx.size)
    tb, rb = helmert(tx.size).T, helmert(rx.size).T
    t, r = tx[ti], rx[ri]
    d = r - t
    R = np.hypot(length, d)
    design = np.column_stack((length/R, (d/R)**2*(length/R), tb[ti], rb[ri]))

    # A nuisance-only fit supplies the initial mask; every channel is tested
    # again against the complete model, including the recovered anomaly.
    threshold = 0.0275
    coef = np.linalg.lstsq(design, voltage, rcond=None)[0]
    keep = np.abs(voltage - design @ coef) < threshold
    u = None
    for _ in range(6):
        profile = _Profile(design, voltage, keep, t, r, length, depth, radius)
        u = profile.search(radius, u)
        x, z = u * [length, depth]
        k = 1./np.sqrt((x*x + (z-t)**2 + radius**2) *
                       ((length-x)**2 + (z-r)**2 + radius**2))
        p = profile.project(k[keep])
        amp = max(0., p @ profile.y) / max(p @ p, 1e-30)
        coef = np.linalg.lstsq(design[keep], (voltage - amp*k)[keep], rcond=None)[0]
        new_keep = np.abs(voltage - design @ coef - amp*k) < threshold
        if np.array_equal(keep, new_keep):
            break
        if _ < 5:
            keep = new_keep
    u_ml = u
    mean = profile.search(radius/2, previous=u_ml, posterior=True)
    if np.linalg.norm((mean-u_ml)*[length,depth]) > .10:
        u = mean
        x, z = u * [length, depth]
        k = 1./np.sqrt((x*x + (z-t)**2 + radius**2) *
                       ((length-x)**2 + (z-r)**2 + radius**2))
        p = profile.project(k[keep])
        amp = max(0., p @ profile.y) / max(p @ p, 1e-30)
        coef = np.linalg.lstsq(design[keep], (voltage - amp*k)[keep], rcond=None)[0]
    offsets = rb @ coef[tx.size+1:]
    return {'anomaly_xz_m': [float(x), float(z)],
            'rx_contact_offset_v': offsets.tolist()}

import numpy as np


def _rosette_features(x, bank):
    a, b, c = x[:, bank], x[:, (bank+1)%4], x[:, (bank-1)%4]
    return np.column_stack((np.ones(len(a)), a, b, c, a*b, a**3))


def _calibrate(study, raw, report=False):
    sweeps = study['sensor_calibration_sweeps']
    x = np.asarray(sweeps['ref_rosette'])
    y = np.asarray(sweeps['raw_rosette'])
    sweeps = sweeps['single_bank_sweeps']
    matrices, readings = [], []
    for j in range(4):
        mask = sweeps['bank_id'] == j
        single = np.zeros((mask.sum(), 4))
        single[:, j] = sweeps['ref_signal'][mask]
        matrices.append(np.vstack((_rosette_features(x,j), _rosette_features(single,j))))
        readings.append(np.r_[y[:,j], sweeps['raw_readout'][mask]])
    def robust(A, b):
        keep = np.ones(len(b), dtype=bool)
        for _ in range(12):
            c = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
            e = np.abs(A@c-b)
            new = e <= max(6*np.median(e),1e-10)
            if np.array_equal(new, keep):
                break
            keep = new
        return np.linalg.lstsq(A[keep],b[keep],rcond=None)[0], keep
    fits = [robust(A,b) for A,b in zip(matrices,readings)]
    mat = np.asarray([c for c,_ in fits]); offset, gain, nxt, prv, cross, cubic = mat.T
    def predict(z):
        return offset + gain*z + nxt*np.roll(z,-1,axis=1) + prv*np.roll(z,1,axis=1) + cross*z*np.roll(z,-1,axis=1) + cubic*z**3
    yy = np.asarray(raw).reshape(-1,4)
    j = np.arange(4)
    jac0 = np.diag(gain)
    jac0[j,(j+1)%4] = nxt
    jac0[j,(j-1)%4] = prv
    z = np.linalg.solve(jac0,(yy-offset).T).T
    for _ in range(20):
        e = predict(z) - yy
        if np.max(np.abs(e)) < 1e-13:
            break
        jac = np.broadcast_to(jac0,(len(z),4,4)).copy()
        jac[:,j,j] += cross*np.roll(z,-1,axis=1) + 3*cubic*z**2
        jac[:,j,(j+1)%4] += cross*z
        z -= np.linalg.solve(jac,e[...,None])[...,0]
    if report:
        print('calibration parameters:',mat,'max inverse error:',np.max(np.abs(predict(z)-yy)))
        for A,b,(c,keep) in zip(matrices,readings,fits):
            err=[]
            for i in np.flatnonzero(keep):
                fold=keep.copy(); fold[i]=False
                err.append(A[i]@np.linalg.lstsq(A[fold],b[fold],rcond=None)[0]-b[i])
            print('calibration worst LOO',max(np.abs(err)), 'outliers',np.flatnonzero(~keep))
    return z.ravel()

def _bridge_design(load):
    n = len(load)
    nxt = np.roll(load, -1, axis=1)
    return np.concatenate((np.ones((n,4,1)), np.broadcast_to(load[:,None,:], (n,4,4)), (load**3)[:,:,None], (load*nxt)[:,:,None]), axis=2)


def _calibrate(calibration):
    raw = calibration['bridge_raw']
    true = calibration['true_channel_loads']
    # The disturbed sweeps are isolated by their large four-channel excursion.
    clean = np.max(np.abs(raw), axis=1) < 8
    design = _bridge_design(true[clean])
    return np.stack([np.linalg.lstsq(design[:,j],raw[clean,j],rcond=None)[0] for j in range(4)])


def _stress(raw, bridge):
    raw = np.asarray(raw)
    measurements = raw.reshape(-1,4)
    offset, linear, cubic, mixed = bridge[:,0], bridge[:,1:5], bridge[:,5], bridge[:,6]
    load = np.linalg.solve(linear, (measurements-offset).T).T
    idx = np.arange(4)
    for _ in range(24):
        nxt = np.roll(load,-1,axis=1)
        error = offset + load @ linear.T + cubic*load**3 + mixed*load*nxt - measurements
        if np.max(np.abs(error)) < 1e-11:
            break
        jac = np.broadcast_to(linear,(len(load),4,4)).copy()
        jac[:,idx,idx] += 3*cubic*load**2 + mixed*nxt
        jac[:,idx,(idx+1)%4] += mixed*load
        step = np.linalg.solve(jac,error[:,:,None])[:,:,0]
        load -= step / np.maximum(1.,np.max(np.abs(step),axis=1,keepdims=True))
    return load.sum(axis=1).reshape(raw.shape[:-1])

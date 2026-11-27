exec(open('/tmp/check_rheology01.py').read().split("print('active'")[0])
out = m.solve_alpha_m97_c03_batch_01(d)
tau = out['estimated_taus']; w = np.r_[out['hereditary_weights'], out['regime_mode_weights']]
# Independent closed-form convolution, using the time elapsed after each past interval.
after = np.cumsum(dt[::-1])[::-1] - dt
z = p @ ((-np.expm1(-dt[:,None]/tau))*np.exp(-after[:,None]/tau))
gap = h[:,None] - z
a0 = w[7]*a*h
slope = w[2] + x @ w[3:7] + w[8]*a*z[:,0]
def curve(t):
    b = -np.expm1(-t[:,None]/tau)
    return (gap*w[:2]) @ b.T + a0[:,None]*b[:,0] + slope[:,None]*np.log1p(t)
rate = (gap*w[:2]) @ (1/tau) + a0/tau[0] + slope
for on in (False, True):
    mask = act[n:] == on
    ec = np.max(abs((curve(t)[n:]+off[n:,None]-out['predicted_strain'])[mask]))
    er = np.max(abs((rate[n:]-out['initial_rate_pred'])[mask]))
    print('identity, damage active',on,'coupon count',mask.sum(),'curve/rate max disagreement',ec,er)
    assert max(ec,er)<1e-11
h0=1e-5
fd = (4*curve(np.array([h0]))[:,0]-curve(np.array([2*h0]))[:,0])/(2*h0)
print('independent zero-time derivative max error',abs(fd-rate).max())
assert abs(fd-rate).max()<1e-8
# Cross-fit by lab so an offset cannot be learned from the lab being scored.
lab=d['train']['lab_id']; complete=~np.isnan(d['train']['coupon_features_raw']).any(1)
for held in np.unique(lab):
    train=lab!=held; obs=(lab==held)&complete
    tf,wf=m._fit((*[v[:n][train] for v in args],dt,t),yy[train].ravel(),'all')
    design=m._basis(np.log(tf),*[v[:n][obs] for v in args],dt,t,'all')[0]
    pred=design@wf; e=pred-yy[obs]
    base=pred-design[:,:,7:]@wf[7:]
    print('held lab',held,'complete n',obs.sum(),'signed median curve/rate',np.round([np.median(e[:,:-1]),np.median(e[:,-1])],6),'median abs full vs omit damage',np.round([np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])),np.median(abs(base[:,:-1]-yy[obs,:-1])),np.median(abs(base[:,-1]-yy[obs,-1]))],6))
print('max relative tau error',abs(tau/g['taus']-1).max(),'max abs hereditary/plateau coefficient error',abs(w[:7]-g['theta']).max(),abs(w[7:]-g['vp_theta']).max())

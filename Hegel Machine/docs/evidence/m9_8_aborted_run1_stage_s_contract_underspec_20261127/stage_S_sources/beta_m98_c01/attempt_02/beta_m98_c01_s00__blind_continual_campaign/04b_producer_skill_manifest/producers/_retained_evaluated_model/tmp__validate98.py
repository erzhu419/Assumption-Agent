from probe98 import *
from scipy.linalg import null_space
key,A,h=next(t for t in candidates() if t[0]==(1,0,2,1)); nom=np.repeat([s['robin_x'],s['robin_y']],2)/h
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; raw=c['raw_rosette']; des=np.c_[np.ones(len(x)),x]; affine=np.linalg.lstsq(des,raw,rcond=None)[0]; cal=lambda y:np.linalg.solve(affine[1:].T,(y.reshape(-1,4)-affine[0]).T).T.ravel(); Y=np.array([cal(p['y_pulse_raw']) for p in p]).T; y=cal(s['y_obs_raw']); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; F=np.column_stack([m.ravel(order='F') for m in s['source_modes']]); n=len(y); ss=len(idx); q=y.reshape(-1,4); fp=lambda p:p.reshape(-1,4); rms=lambda z:np.sqrt(np.mean(z*z));
def design(r):
    AA=A+diags(ends@r,format='csc'); return np.concatenate([splu(AA+mu*E).solve(F)[idx] for mu in s['mu_shifts']]);
W=np.column_stack([a['w_pulse'] for a in p]);
for model, g, x0 in [('nom',lambda u:nom,[]),('axis',lambda u:np.repeat(u,2),nom[::2]),('wall',lambda u:u,nom)]:
    for i in [0,1,None]:
        train=[i] if i is not None else [0,1]; test=[1-i] if i is not None else [0,1]; r=g(least_squares(lambda u:(design(g(u))@W[:,train]-Y[:,train]).ravel(),x0,ftol=1e-13,gtol=1e-13).x) if len(x0) else g(None); R=design(r)@W-Y; print(model,i,'Robin',r*h,'train/test',rms(R[:,train]),rms(R[:,test]),'max-wall-shift',max(rms(a) for a in np.split(R[:,test],3)))
    if model=='wall': r_pulse=r
N=null_space(np.ones((1,W.shape[0]))); bm=np.ones(W.shape[0])*s['total_budget']/W.shape[0]
def obs_fit(r, mask):
    S=design(r); D=np.c_[S@N,Z]; b=y-S@bm; a=np.linalg.lstsq(D[mask],b[mask],rcond=None)[0]; return D@a-b,bm+N@a[:N.shape[1]]
for label,r in [('nom',nom),('pulse',r_pulse)]:
    err,w=obs_fit(r,np.ones(n,dtype=bool)); print(label,'blind residual',rms(err),'weights',w,'pulse/nom sensitivity',np.max(abs(r-nom)))
for k in [0,1,2,None]:
    mask=np.ones(n,dtype=bool); 
    if k is not None: mask[k*ss:(k+1)*ss]=False
    fit=least_squares(lambda r:obs_fit(r,mask)[0][mask],nom,bounds=(.2*nom,3*nom),ftol=1e-12,gtol=1e-12); err,w=obs_fit(fit.x,mask); print('blindfit',k,fit.x*h,'within / out',rms(err[mask]),rms(err[~mask]) if k is not None else 0,'nom/pulse out',[rms(obs_fit(r,mask)[0][~mask]) for r in [nom,r_pulse]] if k is not None else [],'weights',w,'jacobian sv',np.linalg.svd(fit.jac,compute_uv=False))
# Split by rosette so neither channel nor a physical sensor pair leaks between folds.
for fold in range(4):
    tr=np.arange(len(x))%4!=fold; f=np.linalg.lstsq(des[tr],raw[tr],rcond=None)[0]; print('cal',fold,'affine',np.max(np.sqrt(np.mean((des[~tr]@f-raw[~tr])**2,axis=0))),'identity',rms((x-raw)[~tr]))

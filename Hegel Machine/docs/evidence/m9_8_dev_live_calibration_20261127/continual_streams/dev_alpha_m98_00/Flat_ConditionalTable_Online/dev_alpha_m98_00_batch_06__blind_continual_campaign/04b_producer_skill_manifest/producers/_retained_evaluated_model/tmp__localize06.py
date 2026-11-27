import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from chamber06 import *
basey=base@F.T; baser=base@(1/tau); rr=r-baser-dr; sensor=np.r_[np.array([.52,.43])*tau,.05]; f=np.c_[F,t]@sensor
e=y-basey-dr[:,None]*f; ey=e@f/(f@f); print('drift-shape correlation',np.corrcoef(ey,rr)[0,1], 'curve RMS after inferring scalar extra drift from rate',np.sqrt(np.mean((e-rr[:,None]*f)**2))); print('lab stratum missing, drift and residual'); print(np.round(np.c_[lab,g['stratum'],np.isnan(raw).sum(1),dr,rr,ey,np.sqrt(np.mean((e-rr[:,None]*f)**2,axis=1))],4)); print('hold comparison',np.round(np.c_[h,g['strain_0'],baser,rr],4)); print('sweep table',c['single_channel_sweeps']);

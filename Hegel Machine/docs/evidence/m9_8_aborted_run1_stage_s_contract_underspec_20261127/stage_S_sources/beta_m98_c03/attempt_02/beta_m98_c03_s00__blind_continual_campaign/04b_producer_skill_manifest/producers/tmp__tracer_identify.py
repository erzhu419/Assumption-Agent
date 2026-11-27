from transport_probe import *
B,E,hx,hy=init(1,'ar','face','velocity'); d0=np.repeat([rx/hx,ry/hy],2)
ns=len(ids); ng=len(shifts)
# Probe whether the remaining mismatch transfers as a shared transport term.
ax0=axis(s['kappa_x'],s['vel_x'],hx,'ar','face','none'); ay0=axis(s['kappa_y'],s['vel_y'],hy,'ar','face','none')
ax=axis(s['kappa_x'],s['vel_x'],hx,'ar','face','velocity'); ay=axis(s['kappa_y'],s['vel_y'],hy,'ar','face','velocity')
D=np.kron(np.eye(ny),ax0)+np.kron(ay0,np.eye(nx)); A=np.kron(np.eye(ny),ax-ax0)+np.kron(ay-ay0,np.eye(nx)); V=np.diag(s['V_reaction_2d'].ravel(order='F'))
Bface,*_=init(1,'ar','face','facevelocity'); Bflux,*_=init(1,'ar','face','flux'); Bharm,*_=init(1,'ha','face','velocity')
folds=[]
for p in range(y.shape[1]):
 test=np.zeros_like(y,dtype=bool); test[:,p]=True; folds.append(('pulse',test))
for k in range(ng):
 test=np.zeros_like(y,dtype=bool); test[k*ns:(k+1)*ns]=True; folds.append(('shift',test))
for k in range(4):
 test=np.broadcast_to(((np.arange(len(y))%ns)//4%4==k)[:,None],y.shape); folds.append(('rosette',test))
for name,mat in [('walls only',None),('diffusion scale',D),('velocity scale',A),('reaction scale',V),('velocity indexing',Bface-B),('conservative flux',Bflux-B),('diffusion averaging',Bharm-B)]:
 def pred(a): return predict(B if mat is None else B+a[-1]*mat,E@a[:4])
 ini=d0 if mat is None else np.r_[d0,0.]; lower=(0.,)*4 if mat is None else (0.,)*4+(-.4,); upper=(np.inf,)*4 if mat is None else (np.inf,)*4+(.4,)
 full=least_squares(lambda a:(pred(a)-y).ravel(),ini,bounds=(lower,upper),gtol=1e-12)
 results={}; corrections=[]
 for kind,test in folds:
  fit=least_squares(lambda a:(pred(a)-y)[~test],ini,bounds=(lower,upper),gtol=1e-12)
  err=(pred(fit.x)-y).reshape(ng,ns,-1); mask=test.reshape(err.shape)
  slices=[np.sqrt(np.mean(err[k,:,p][mask[k,:,p]]**2)) for k in range(ng) for p in range(y.shape[1]) if mask[k,:,p].any()]
  results[kind]=max(results.get(kind,0.),max(slices)); corrections.append(fit.x[-1])
 print(name,'full RMSE',np.sqrt(np.mean(full.fun**2)), 'worst by holdout',results,'coefficient range', (min(corrections),max(corrections)) if mat is not None else None)

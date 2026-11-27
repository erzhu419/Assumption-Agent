exec(open('/tmp/rheo_clock.py').read().split("print('nominal drift")[0])
from scipy.optimize import least_squares
rms=lambda a:np.sqrt(np.mean(a*a))
L=np.r_[np.log1p(t),1.]
print('exact reset + log1p; columns fit curve/rate, worst lab curve/rate, worst time, complete/missing pooled')
for name,f in variants.items():
 E=hold+dr(f)[:,None]*L-y
 print(name,np.round([rms(E[:,:-1]),rms(E[:,-1]),max(rms(E[lab==g,:-1]) for g in labs),max(rms(E[lab==g,-1]) for g in labs),max(rms(E[:,j]) for j in range(len(t))),rms(E[clean]),rms(E[~clean])],6))
F=variants['diag_only']; R=dr(F); delta=y-hold
print('exact per-lab curve/rate rmse',[(g,rms((R[:,None]*L-delta)[lab==g,:-1]),rms((R[:,None]*L-delta)[lab==g,-1])) for g in labs])
print('diagnostic drift differences per row',np.round(np.c_[R,(delta@L)/(L@L)-R,delta[:,-1]-R,clean],5))
for g in np.r_[-1,labs]:
 keep=lab!=g; test=~keep if g>=0 else keep
 def fit(v):
  k,a=v[:2]; G=np.r_[np.log1p(t/k),1/k]; return delta-a*R[:,None]*G+(M*v[2:])@B
 out=least_squares(lambda v:fit(v)[keep].ravel(),[1,1,0,0],bounds=([.05,0,-2,-2],[20,20,2,2])); E=fit(out.x)
 print('free log timescale, gain, memory; heldout RMSE curve/rate',g,out.x.round(5),rms(E[test,:-1]),rms(E[test,-1]))
# Compare normalization hypotheses on labeled benchtop specimens without moving fitted material parameters.
for name,X in [('joint',x[:n]),('train alone',m._features(tr,tr)[:n])]:
 a,r=m._design(X,*m._loads(tr,b),dt,t,tau); ec=(a@th).reshape(tr['strain_curves_obs'].shape)+tr['strain_0'][:,None]-tr['strain_curves_obs']; er=r@th-tr['instant_rate_obs']; print('train using',name,'median absolute curve/rate and rate RMS',np.median(abs(ec)),np.median(abs(er)),rms(er))

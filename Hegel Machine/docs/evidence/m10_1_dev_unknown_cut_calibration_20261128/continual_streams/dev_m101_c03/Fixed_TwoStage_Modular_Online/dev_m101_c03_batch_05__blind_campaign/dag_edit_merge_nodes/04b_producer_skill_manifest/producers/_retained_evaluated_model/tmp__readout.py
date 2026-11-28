from cal import *
from panels import poly
from scipy.optimize import least_squares
x=s['train']['x']; z=invert(x); b=forward(z); bi=backward(z)
Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; n=len(x); print('N',n)
idx=n-20; rng=np.random.default_rng(17)
folds=[(np.arange(idx),np.arange(idx,n))]+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(rng.permutation(n),5)]
def scores(m,alpha=0):
 E=[]
 for i,j in folds:
  S=np.linalg.norm(m[i],axis=0); S=np.maximum(S,1e-10); a=m[i]/S
  w=np.linalg.lstsq(a,Y[i],rcond=1e-9)[0] if not alpha else np.linalg.solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@Y[i]); e=abs((m[j]/S)@w-Y[j]); E.append((e[:,0].mean(),e[:,1:].mean()))
 return np.round([*E[0],*np.max(E,axis=0)],5)
for k,a in {'x':x,'z':z,'b':b,'bi':bi}.items():
 for deg in (2,3,4,5):
  m,names=poly(a,deg); fit=m@np.linalg.lstsq(m,Y,rcond=None)[0]; print(k,deg,'train',np.mean(abs(fit-Y),axis=0)[[0,-1]].round(5),'CV',[(l,scores(m,l)) for l in (0,1e-5,1e-3)])
 m,names=poly(a,4); print(k,'Y cubic coeffs > .015',[(j,round(c,4)) for j,c in zip(names,np.linalg.lstsq(m,Y,rcond=None)[0][:,0]) if abs(c)>.015])
 for form,f in {'z-squares':np.c_[a,a*a,q(a)],'panel-readouts':np.c_[a,q(a),h(a)],'C1C2-mixed':np.c_[a,q(a),h(a),q(a)*h(a),q(a)*a,q(a)*np.roll(a,-1,axis=1),q(a)**2],'C1C2-product':np.c_[a,q(a),h(a),(lin(q(a))@C)*(lin(h(a))@D)]}.items(): print(k,form,scores(lin(f),1e-6))
# Estimate temporal basis before fitting the spatial amplitudes.
for name,f in {'exp+t':lambda k:np.c_[1-np.exp(-k*t),t], 'exp+integral':lambda k: np.c_[1-np.exp(-k*t),t-(1-np.exp(-k*t))/k], 'exp+decay':lambda k:np.c_[1-np.exp(-k*t),np.exp(-k*t)],'exp+texp':lambda k:np.c_[1-np.exp(-k*t),t*np.exp(-k*t)]}.items():
 def resid(p):
  m=f(np.exp(p[0])); return (Y[:,1:]-np.linalg.lstsq(m,Y[:,1:].T,rcond=None)[0].T@m.T).ravel()
 r=least_squares(resid,[0.]); print('time',name,np.exp(r.x),np.mean(abs(r.fun)))

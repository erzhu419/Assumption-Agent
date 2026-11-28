from discover import *
from scipy.optimize import least_squares, minimize
p=s['panel_b']; u,v=p['u'],p['v']; X=cal(s['train']['x'],True); Y=s['train']['y']; T=s['train']['traj']; t=s['eval_times'];
def bf(x,w,kind):
 a=np.c_[one(x),np.sin(w*x)]
 if kind=='cos': return np.c_[a,np.cos(w*x)-1]
 if kind=='sq': return np.c_[a,x*x]
 if kind=='cycle': return np.c_[a,x*np.roll(x,-1,axis=1)]
 if kind=='pairs': return np.c_[a,x[:,0]*x[:,3],x[:,1]*x[:,2]]
 return a

def panel(u,v,kind,shared):
 def resid(w):
  a=bf(u,w,kind);return (a@fit(a,v)-v).ravel()
 w=least_squares(resid,[2.8]*(1 if shared else 4),bounds=(1.,4.)).x
 return w,fit(bf(u,w,kind),v)
for kind in ['plain','cos','sq','cycle','pairs']:
 for shared in [True,False]:
  w,c=panel(u,v,kind,shared); r=bf(u,w,kind)@c-v; e=[]
  for i in range(len(u)):
   g=np.arange(len(u))!=i; w0,c0=panel(u[g],v[g],kind,shared);e.append(bf(u[~g],w0,kind)@c0-v[~g]);
  print('B',kind,shared,w.round(4),np.sqrt(np.mean(r*r)),np.sqrt(np.mean(np.array(e)**2)))
  if kind=='plain' and shared: print('coeff',c.round(4)); print('residual',r.round(4)); Z=bf(X,w,kind)@c
print('times',t); us,sv,vs=np.linalg.svd(T,full_matrices=False); S=T@vs[0]; print('state rate',least_squares(lambda p: -p[0]*np.expm1(-p[1]*t)-vs[0],[.7,.5]).x,'basis',vs[0]);
for nm,a in [('Z',Z),('X',X),('both',np.c_[Z,X])]:
 R=np.c_[Y,S]-one(a)@fit(one(a),np.c_[Y,S]); print(nm,'linear coef',fit(one(a),np.c_[Y,S]).round(4),'rmse',np.std(R,axis=0),'range',np.quantile(R,[0,.1,.5,.9,1],axis=0).round(4))
 if nm!='Z':continue
 j,k=np.triu_indices(4); pairs=Z[:,j]*Z[:,k];
 print('quad coefficients',fit(np.c_[one(Z),pairs],np.c_[Y,S]).round(3),'pairs',list(zip(j,k)))
 F=np.c_[pairs,Z**3,Z**5,np.sin(Z),np.cos(Z),X*X,np.sin(2.8*X),*(pairs/(1+a*Z[:,q:q+1]**2) for a in [.5,1,2] for q in range(4))]; D=one(Z); P=F-D@fit(D,F); print('best residual-correlated features',[(i,[(int(j),round(float(P[:,j]@R[:,i]/np.linalg.norm(P[:,j])/np.linalg.norm(R[:,i])),3)) for j in np.argsort(np.abs(P.T@R[:,i])/np.linalg.norm(P,axis=0))[-12:][::-1]]) for i in range(2)])
 for i in range(2):
  for h in [0,1,2,3]:
   ix=np.argsort(Z[:,h]); print('res',i,h,np.round(np.c_[Z[ix,h],R[ix,i]][::5].T,2))
print('cal reconstruction',np.max(abs(cal(X)-s['train']['x'])))

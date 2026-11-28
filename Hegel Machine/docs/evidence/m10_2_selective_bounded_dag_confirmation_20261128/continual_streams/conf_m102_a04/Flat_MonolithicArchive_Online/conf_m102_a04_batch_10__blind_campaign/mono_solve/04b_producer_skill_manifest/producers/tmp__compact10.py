exec(open('/tmp/audit10.py').read().split('for F,tar')[0])
t=s['eval_times']; diag=(g*h).sum(1); off=g[:,0]*h[:,1]; features=[D,m.one(np.c_[g,h,diag,off]),m.one(np.c_[g[:,0]+g[:,2],g[:,1],g[:,3],h,diag,off])]; groups=[np.arange(52,72),np.argsort(np.linalg.norm(z,axis=1))[-20:], *(np.random.default_rng(i).permutation(n)[:20] for i in range(8))]; dest=[]
for fi,F in enumerate(features):
 vals=[]
 for test in groups:
  fit=np.setdiff1d(np.arange(n),test); y=Y[fit,1:]; amp=m.lin(F[fit],y); b=m._time_basis(t,y); pred=[F[test]@amp,F[test]@m.lin(F[fit],m.lin(b,y.T).T)@b.T]; vals.append([[np.abs(p-Y[test,1:]).mean(),np.abs(p-Y[test,1:]).mean(0).max()] for p in pred]);
 vals=np.asarray(vals); print('grammar',fi,'public',vals[0], 'max',vals.max(0),'mean',vals.mean(0));
 if fi==2:
  print('coeff time matrix',np.round(m.lin(F,Y[:,1:]),4)); print('coefficient SVD',np.linalg.svd(m.lin(F,Y[:,1:]),compute_uv=False))
p=s['panel_c']; u=p['u']; y=p['v1']; from scipy.optimize import least_squares
for f,k in [(lambda u,k:np.tanh(k*u),1.77),(lambda u,k:u/np.sqrt(1+k*u*u),1.60),(lambda u,k:u/(1+k*u*u),1.26)]:
 def d(x,k):return m.one(np.c_[f(x,k),x,x**3,x**5])
 e=[]
 for idx in range(len(u)):
  keep=np.arange(len(u))!=idx; f1=lambda p:(d(u[keep],p[0])@m.lin(d(u[keep],p[0]),y[keep])-y[keep]).ravel(); q=least_squares(f1,[k],bounds=(k*.8,k*1.2),gtol=1e-12).x[0]; e.append(d(u[[idx]],q)@m.lin(d(u[keep],q),y[keep])-y[[idx]])
 print('panel loo',np.sqrt(np.mean(np.asarray(e)**2)),np.max(np.abs(e)))

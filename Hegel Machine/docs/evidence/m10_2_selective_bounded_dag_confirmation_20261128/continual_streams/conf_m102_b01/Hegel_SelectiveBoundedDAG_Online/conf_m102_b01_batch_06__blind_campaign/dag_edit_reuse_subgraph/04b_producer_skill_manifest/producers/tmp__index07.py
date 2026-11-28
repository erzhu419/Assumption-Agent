exec(open('/tmp/fit07.py').read().split('for name,b')[0])
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=150)
W=w[:n]; R=W*np.roll(W,-1,axis=1); Q=np.linalg.lstsq(np.c_[np.ones(len(u)),u*np.roll(u,-1,axis=1)],p['v1'],rcond=None)[0]; V=np.c_[np.ones(n),R]@Q
print('panel B ranges',[(k,np.min(v,axis=0),np.max(v,axis=0)) for k,v in s['panel_b'].items()]); print('w test ranges',w[n:].min(0),w[n:].max(0)); print('ring fit outputs', np.linalg.lstsq(np.c_[np.ones(n),R,W],np.c_[y,tr],rcond=None)[0]); print('time singular',np.linalg.svd(tr,full_matrices=False)[2][:3]);
Z=np.c_[R,W]; AA=np.c_[np.ones(n),Z]; Z=Z/np.std(Z,axis=0); PC=np.linalg.svd(tr,full_matrices=False)[2][:3].T
rng=np.random.default_rng(8)
for name,B in [('y',y[:,None]),('tr',tr@PC),('joint',np.c_[y,tr@PC])]:
 opts=[]
 for space,ZZ in [('ring',Z[:,:4]),('all',Z),('w',Z[:,4:])]:
  for bas,F in [('ring',AA[:,:5]),('all',AA),('only',AA[:,:1]),('squares',np.c_[AA,R*R])]:
   for kind in ['tanh','sin','rat','exp','den','mix','soft','abs']:
    def fit(a,v):
     Y=B[a]; sd=np.maximum(Y.std(0),.10); d=ZZ.shape[1];
     def mat(q,i):
      z=ZZ[i]@q[:d]+q[d]; f=F[i];
      if kind=='den':return f/(1+z[:,None]**2)
      if kind=='mix':return np.c_[f,f/(1+z[:,None]**2)]
      h={'tanh':np.tanh,'sin':np.sin,'rat':lambda z:z/(1+z*z),'exp':lambda z:np.exp(-z*z),'soft':lambda z:np.sqrt(1+z*z),'abs':np.abs}[kind](z)
      return np.c_[f,h]
     def fun(q):
      M=mat(q,a); r=M@np.linalg.lstsq(M,Y,rcond=1e-8)[0]-Y;return np.r_[(r/sd).ravel(),.003*q]
     q=min([least_squares(fun,x,max_nfev=95) for x in [rng.normal(0,f,d+1) for f in [.15,.55,1.2]]],key=lambda r:r.cost).x
     M=mat(q,a); coef=np.linalg.lstsq(M,Y,rcond=1e-8)[0]; pred=mat(q,v)@coef
     e=abs((pred-B[v])@PC.T).mean() if name=='tr' else (abs(pred[:,0]-B[v,0]).mean() if name=='y' else (abs(pred[:,0]-B[v,0]).mean()+abs((pred[:,1:]-B[v,1:])@PC.T).mean())/2)
     return e,q,np.mean(abs(M@coef-Y))
    a,v=folds[0]; e,q,te=fit(a,v); opts.append((e,te,space,bas,kind,q));
    if e<.043: print('promising',name,e,te,space,bas,kind,q,flush=True)
 print(name,'first split best',*sorted(opts,key=lambda z:z[0])[:8],sep='\n',flush=True)

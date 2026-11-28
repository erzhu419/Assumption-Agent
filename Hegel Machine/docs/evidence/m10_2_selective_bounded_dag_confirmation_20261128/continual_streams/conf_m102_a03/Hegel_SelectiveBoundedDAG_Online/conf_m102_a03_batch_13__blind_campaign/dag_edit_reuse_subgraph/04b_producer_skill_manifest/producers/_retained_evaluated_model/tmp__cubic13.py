from read13 import *
from scipy.optimize import least_squares
A=aff(b); Q=poly(b,2,2); C=poly(b,3,3); F=np.c_[A,Q,C]; labels=[p for k in (2,3) for p in it.combinations_with_replacement(range(4),k)]; c=fit(F,y); print('coeffs',[(p,round(float(j),4)) for p,j in zip(labels,c[5:])]); print('train cubic MAE',abs(F@c-y).mean()); print('cubic CV',scores(F,y,.0001));r=b[:,0]*b[:,2]+b[:,1]*b[:,3]; base=np.c_[A,r]; bf0=fit(base,y); print('base residual MAE',abs(base@bf0-y).mean())
for k in (1,2,3,4,5,6,8,12,20):
 e=[]; chosen=[]
 for i,j in splits:
  D=np.c_[Q,C]; ids=pursuit(base[i],D[i],y[i,None],k,()); G=np.c_[base,D[:,ids]]; e.append(abs(G[j]@fit(G[i],y[i])-y[j]).mean()); chosen.append([labels[l] for l in ids]);
 print('sparse',k,np.round([e[0],np.median(e),max(e)],5),chosen[0])
for fun in (np.sin,np.tanh,lambda x:x/(1+x*x),lambda x:x**3):
 e=[]; e0=[]
 for i,j in splits:
  a=base[i]; q=np.linalg.qr(a)[0]; yy=y[i]-q@(q.T@y[i]);
  def prof(h):
   v=fun(h[0]+b[i]@h[1:]);v-=q@(q.T@v);return yy-v*(v@yy)/max(v@v,1e-20)
  rng=np.random.default_rng(341); starts=np.r_[np.c_[np.zeros(4),np.eye(4)],rng.normal(size=(20,5))]; o=min((least_squares(prof,h,max_nfev=110) for h in starts),key=lambda o:o.cost); G=np.c_[base,fun(o.x[0]+b@o.x[1:])]; cc=fit(G[i],y[i]);e.append(abs(G[j]@cc-y[j]).mean()); e0.append((o.x.round(3),round(float(cc[-1]),3),round(float(np.mean(abs(o.fun))),5)))
 print('single-index',fun,np.round([e[0],np.median(e),max(e)],5),e0)

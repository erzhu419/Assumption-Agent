from b05_probe import *
from itertools import combinations
u,v=(S['panel_b'][k] for k in ('u','v'))

def warp(x,p,k):
 f,a=p[:2]; w=x+a*np.sin(f*x)
 if k=='cube':w+=p[2]*x**3
 if k=='harmonic':w+=p[2]*np.sin(2*f*x)
 if k=='rational':w+=p[2]*x/(1+p[3]*x*x)
 if k=='tanh':w+=p[2]*np.tanh(p[3]*x)
 if k=='nested':w=x+a*np.sin(f*x+p[2]*x**3)
 if k=='frequency':w=x+np.sin(x*np.r_[f,p[2:5]])*a
 if k=='cross':w+=p[2]*x*np.roll(x,-1,axis=1)
 if k=='after':w+=p[2]*w*np.roll(w,-1,axis=1)
 if k=='cos':w+=p[2]*(np.cos(f*x)-1.)
 return np.c_[np.ones(len(x)),w]

def learn(k,p,i):
 def residual(p):
  a=warp(u[i],p,k);return (a@fit(a,v[i])-v[i]).ravel()
 lo=np.array([1.5,-1]+([-2]*(len(p)-2)));hi=np.array([4.5,-.05]+([4]*(len(p)-2)))
 if k in ('rational','tanh'):lo[-1]=.1
 p=least_squares(residual,p,bounds=(lo,hi),max_nfev=100).x
 return p,fit(warp(u[i],p,k),v[i])

t=tr['traj']; y=tr['y']; tt=S['eval_times']; folds=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]; rng=np.random.default_rng(148); folds += [(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(rng.permutation(n),4)]; folds += [(np.setdiff1d(np.arange(n),j),j) for j in [np.argsort(np.max(np.abs(za[:n]),axis=1))[-16:]]];
for k,start in [('base',[2.85,-.4]),('cube',[2.85,-.4,0]),('harmonic',[2.85,-.4,0]),('rational',[2.85,-.4,0,1]),('tanh',[2.85,-.4,0,1]),('nested',[2.85,-.4,0]),('frequency',[2.85,-.4,2.85,2.85,2.85]),('cross',[2.85,-.4,0]),('after',[2.85,-.4,0]),('cos',[2.85,-.4,0])]:
 p,b=learn(k,start,np.arange(24)); pe=[]
 for j in np.array_split(np.random.default_rng(99).permutation(24),4):
  pp,bb=learn(k,start,np.setdiff1d(np.arange(24),j));pe.extend((warp(u[j],pp,k)@bb-v[j]).ravel())
 z=warp(za,p,k)@b; a=poly(z,1)[0];q=poly(z,2)[0];sparse=np.c_[a,z[:,0]*z[:,2],z[:,1]*z[:,3]]; full=np.c_[sparse,z[:,0]*z[:,3],z[:,2]*z[:,3]]
 print(k,'p',np.round(p,6),'panel rms/CV',np.std(warp(u,p,k)@b-v),np.std(pe))
 for label,g in [('sparse',sparse),('full',full),('quad',q)]:
  yp=[];tp=[]
  for i,j in folds:
   yp.append(np.mean(abs(g[j]@fit(g[i],y[i])-y[j])));tp.append(np.mean(abs(a[j]@fit(a[i],t[i])-t[j])))
  print(label,np.round(yp,5),np.round(tp,5))
 if k=='base':
  for hz in ([1.7,.2],[1.7,.1],[1.7,.3],[1.8,.2]):
   T=-np.expm1(-tt[:,None]*np.array(hz));print('rates',hz,'rms',np.std(T@fit(T,t.T)-t.T))

from explore import *
np.set_printoptions(precision=5,suppress=True)
u=s['panel_c']['u']; v=s['panel_c']; bp=lin(one(pb['u']),pb['v']); print('B',bp)
def maker(y,g,init):
 def fun(q):
  f=g(u,q[0]);return (f@lin(f,y)-y).ravel()
 q=least_squares(fun,[init],bounds=(.04,3),gtol=1e-13).x[0]; F=g(u,q); co=lin(F,y); print('panel q, co, rmse',q,co,np.std(fun([q])));return lambda x:g(x,q)@co
c=maker(v['v1'],lambda x,q:one(np.c_[x,x/(1+q*abs(x))]),.85)
e=maker(v['v2'],lambda x,q:one(np.c_[x,x*x,x*x/(1+q*x*x)]),.3)
B=lambda x:one(x)@bp
from itertools import combinations
pa=s['panel_a']; ua,va=pa['u'],pa['v']; m=np.linalg.norm(va,axis=1)<4; print('cal retained',m.sum());
for deg in [1,2,3,4]:
 p=PolynomialFeatures(deg); f=p.fit_transform(ua); coef=lin(f[m],va[m]); print('cal poly',deg,np.std(f[m]@coef-va[m]));
 if deg in [2,3]:
  for i in range(4):print(i,[(n,round(coef[j,i],5)) for j,n in enumerate(p.get_feature_names_out()) if abs(coef[j,i])>.007])
for i in range(4):
 j=(i+1)%4; xx=ua[:,[i,j]]; p=PolynomialFeatures(4); f=p.fit_transform(xx); coef=lin(f[m],va[m,i]); print('local',i,np.std(f[m]@coef-va[m,i]),np.round(coef,5));
print('clean norm',np.sort(np.linalg.norm(va,axis=1))[-10:]);
p=PolynomialFeatures(2); f=p.fit_transform(ua); coef=lin(f[m],va[m]);
def inv(raw):
 return np.array([least_squares(lambda z:p.transform(z[None,:])[0]@coef-w, (w-o)/k,gtol=1e-11).x for w in raw]);
Z=inv(X); W=B(Z); print('Z ranges',Z.min(0),Z.max(0));
sub=np.arange(52); te=np.arange(52,72); Pairs=list(zip(*np.triu_indices(4,1))); rng=np.random.default_rng(21)
splits=[(sub,te)]+[(p[:52],p[52:]) for p in [rng.permutation(N) for _ in range(3)]]
D=np.c_[Y,T]; results=[]
def check(label,p,regs=(.001,.1,1)):
 for reg in regs:
  vals=[]; fits=[]
  for tr,ts in splits:
   mu=p[tr].mean(0); sd=np.maximum(p[tr].std(0),1e-7); Q=(p-mu)/sd; Q=np.c_[np.ones(N),Q]; K=Q@Q.T; pred=K[:,tr]@np.linalg.solve(K[np.ix_(tr,tr)]+reg*np.eye(len(tr)),D[tr]); err=abs(D-pred); vals.append([err[ts,0].mean(),err[ts,1:].mean()]); fits.append([err[tr,0].mean(),err[tr,1:].mean()]);
  vals=np.array(vals); results.append((label,reg,vals[0],vals.mean(0),np.mean(fits,0)))
for name,x in [('Z',Z),('W',W),('raw',X),('Binv',(Z-bp[0])@np.linalg.inv(bp[1:]))]:
 a1,a2=c(x),e(x); stack=np.c_[a1,a2]; pr=(a1[:,:,None]*a2[:,None,:]).reshape(N,-1); sq=np.c_[a1*a1,a2*a2]; intra=np.column_stack([q[:,i]*q[:,j] for q in (a1,a2) for i,j in Pairs]);
 for nm,ft in [('lin',stack),('pair',np.c_[stack,a1*a2]),('dot',np.c_[stack,(a1*a2).sum(1)]),('cross',np.c_[stack,pr]),('full2',np.c_[stack,pr,sq,intra]),('sq',np.c_[stack,a1*a2,sq]),('intra',np.c_[stack,sq,intra]),('xlin',np.c_[stack,x]),('xpair',np.c_[stack,x,a1*a2])]:check(name+nm,ft)
 for nm,trans in [('bc',B(a1)),('be',B(a2)),('cb+eb',B(a1)+B(a2)),('csum',a1+a2),('cdiff',a1-a2)]:
  check(name+nm,np.c_[stack,trans,trans**2],(.001,.1))
 for q in [.2,.5,1.,1.25]:
  for nm,tr in [('rat',a1/(1+q*a2**2)),('rat2',a2/(1+q*a1**2)),('tanh',np.tanh(q*stack)),('tanhcross',np.tanh(q*(a1+a2))),('soft',stack/(1+q*abs(stack)))]:check(name+nm+str(q),np.c_[stack,a1*a2,tr],(.001,.1))
for k in [0,1]:
 print('BEST',k)
 for r in sorted(results,key=lambda z:z[3][k])[:17]:print(r)

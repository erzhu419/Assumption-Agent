exec(open('/tmp/readout.py').read().split('for nam,a0')[0])
U,S,V=np.linalg.svd(T,full_matrices=False); L=U[:,:3]*S[:3]; Y=np.c_[y,L]; scale=np.array([1,1,.1,.1]);
from sklearn.linear_model import Ridge
from itertools import combinations
b=af(polys(w,2));
def check(tag,Q):
 Q=af(Q); m=Q[ix].mean(0); m[0]=0; sc=Q[ix].std(0).clip(1e-8); sc[0]=1; Q=(Q-m)/sc
 out=[]
 for a in (1e-9,.0001,.003,.03,.2):
  coef=np.linalg.solve(Q[ix].T@Q[ix]+a*np.eye(Q.shape[1]),Q[ix].T@Y[ix]); P=Q@coef; er=np.mean(abs(P[ix]-Y[ix]),0); ev=np.mean(abs(P[val]-Y[val]),0); st=np.mean(abs((P[val,1:]-L[val])@V[:3])); out.append((ev[0],st,a,er[0],ev[1]));
 return sorted(out)[0]
for name,Q in [('b',b[:,1:]),('b+z',np.c_[b[:,1:],z]),('b+z2',np.c_[b[:,1:],polys(z,2)]),('hz2',polys(np.c_[h,z],2)),('hz3',polys(np.c_[h,z],3)),('b+hz',np.c_[b[:,1:],z,(h[:,:,None]*z[:,None,:]).reshape(len(z),16)]),('b+h*z',np.c_[b[:,1:],h*z]),('b+w*z',np.c_[b[:,1:],w*z]),('b+z+h*z',np.c_[b[:,1:],z,h*z]),('b+z+w*z',np.c_[b[:,1:],z,w*z]),('b+z+w*z+z2',np.c_[b[:,1:],z,w*z,z*z])]: print('E',name,check(name,Q))
# Build one-term residual grammar with exact symmetries and mixed coordinates.
prev=np.c_[b,z]; vlist=[]; names=[]
def add(a,n):
 a=np.array(a).reshape(len(z),-1)
 for j,col in enumerate(a.T):
  if col.std()>1e-9:vlist.append(col); names.append(n+str(j))
for nm,x0 in [('w',w),('z',z),('h',h),('w+z',w+z),('w-z',w-z),('w*z',w*z),('w*h',w*h)]:
 add(polys(x0,3),nm+'poly');
 for s0 in (.25,.5,1,1.25,2,3,4,6):
  for ff in (np.sin,np.cos,np.tanh,abs,lambda x:np.maximum(x,0),lambda x:x/(1+x*x),lambda x:1/(1+x*x),lambda x:np.sign(x)*np.maximum(abs(x)-.22,0)**1.25):
   add(ff(s0*x0),nm+str(s0)+ff.__name__)
 for i,j in combinations(range(4),2):
  for sgn in (-1,1):
   v=x0[:,i]+sgn*x0[:,j]; n=f'{nm}{i}{sgn:+d}{j}'
   for f in (abs,np.tanh,np.sin,np.cos,lambda x:x/(1+x*x),lambda x:np.maximum(x,0)**2):add(f(v),n+f.__name__)
for p in ((0,1,2,3),(0,2,1,3),(0,3,1,2)):
 for sg in (-1,1):
  v=w[:,p[0]]*w[:,p[1]]+sg*w[:,p[2]]*w[:,p[3]]
  for ff in (abs,np.tanh,np.sin,np.cos,lambda x:x/(1+x*x),lambda x:x**2):add(ff(v),str(p)+str(sg)+ff.__name__)
for nm,f in [('C',__import__('conf_m102_a03_batch_02_solver')._panel_operators(s['panel_c'])[0]),('D',__import__('conf_m102_a03_batch_02_solver')._panel_operators(s['panel_c'])[1])]:
 for arg,nn in [(z,'z'),(w,'w'),(h,'h')]:
  u=f(arg); add(polys(u,2),nm+nn+'poly'); add(u*w,nm+nn+'*w'); print('E aux',nm,nn,check(nm,np.c_[b[:,1:],z,u,polys(u,2)[:,4:]]))
F=np.array(vlist).T
for pn,pr in [('quad',b),('quad+z',prev)]:
 rr=Y[ix]-pr[ix]@fit(pr[ix],Y[ix]); f=F[ix]-pr[ix]@fit(pr[ix],F[ix]); qos=(f.T@rr/scale)**2/(np.sum(f*f,axis=0)[:,None]+1e-14); best=np.argsort(-qos.sum(1))[:12];print('top',pn,[(names[j],round(qos[j].sum(),5),check('m',np.c_[pr[:,1:],F[:,j]])) for j in best]);
 for k in range(8):
  rr=Y[ix]-pr[ix]@fit(pr[ix],Y[ix]); f=F[ix]-pr[ix]@fit(pr[ix],F[ix]); js=np.sum((f.T@rr/scale)**2,axis=1)/(np.sum(f*f,axis=0)+1e-12); j=js.argmax(); pr=np.c_[pr,F[:,j]]; print('step',k,names[j],check('m',pr[:,1:]));

from f import *
P=s['panel_c'];u=P['u']; V=P['v2']; ix=[(0,1),(0,3),(1,2),(2,3)];
def Q(a): return np.column_stack([np.ones(len(a))]+[a[:,i]*a[:,j] for i,j in ix]);
c=ls(Q(u),P['v1']); print('C1',c,'res',np.max(abs(Q(u)@c-P['v1'])));
B=lambda p:A(p)@b; C=lambda p:Q(p)@c
Y=np.c_[y,T];
def score(f):
 e=abs(f@ls(f[:52],Y[:52])-Y);return np.r_[np.mean(e[:52,0]),np.mean(e[52:,0]),np.mean(e[:52,1:]),np.mean(e[52:,1:])]
layers={'q':q,'z':z,'zi':zi}; new={k:'unused' for k in ()}
for k,p in list(layers.items()):
 for h,op in [('C',C),('B',B)]:
  r=op(p);layers[h+k]=r;layers[h+'2'+k]=op(r)
  if h=='C': layers['BC'+k]=B(r);layers['CBC'+k]=C(B(r));layers['CCB'+k]=C(B(p))
res=[]
for k,p in layers.items():
 for h,r in [('alone',p),('q',np.c_[q,p]),('z',np.c_[z,p]),('Cq',np.c_[C(q),p])]:
  for d in (1,2):
   f=poly(r,d);f=f if f.shape[1]<52 else f[:,:52];res.append((k,h,d,score(f)))
 for h,r in [('qp',q*p),('zp',z*p),('Cqp',C(q)*p)]:res.append((k,h,1,score(A(np.c_[q,p,r]))))
for j in range(4):print('rank',j, sorted(res,key=lambda v:v[-1][j])[:12]);

def models(p,k):
 a=p*p
 return [np.abs(p),a,a*a,a*a*a,a/(1+k*a),1/(1+k*a),np.abs(p)/(1+k*np.abs(p)),np.sqrt(1+k*a),np.log1p(k*a),np.cos(k*p),np.cos(k*a),np.sin(k*a),np.exp(-k*a),np.tanh(k*a),a/(1+k*np.abs(p)),a*a/(1+k*a),a/(1+k*a*a),np.tanh(k*p),np.sin(k*p),p/(1+k*a),np.abs(p)**k,np.sqrt(abs(p)+k),np.sqrt(a+k),np.maximum(a-k,0),np.maximum(abs(p)-k,0)]
for j in range(len(models(u,1))):
 for base in (lambda p:np.ones((len(p),1)), Q, A):
  def f(p,k):return np.c_[base(p),models(p,k)[j]]
  def fun(k):
   F=f(u,k[0]);return (F@ls(F,V)-V).ravel()
  for k0 in (.22,1.25,3.):
   o=least_squares(fun,[k0],bounds=(.01,10),max_nfev=100); a,l=loo(f(u,o.x[0]),V);
   if k0==.22 or np.mean(a)<.005: print('C2',j,base.__name__,o.x,np.mean(a),np.mean(l))

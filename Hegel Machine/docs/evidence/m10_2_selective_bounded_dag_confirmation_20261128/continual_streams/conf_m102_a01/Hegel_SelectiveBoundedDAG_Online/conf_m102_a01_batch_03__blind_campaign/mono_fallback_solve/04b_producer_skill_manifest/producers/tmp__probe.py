from explore import *
from sklearn.model_selection import KFold
from scipy.linalg import solve
G={'x':x,'z':z,'h':h,'Bz':z@B[1:]+B[0]}; C=fit(aff(c['u']),c['v1']); G['Cz']=z@C[1:]+C[0]; G['Ch']=h@C[1:]+C[0]
ff=np.c_[np.ones(len(z)),z]; folds=[(np.arange(52),np.arange(52,72))]+list(KFold(5,shuffle=True,random_state=21).split(np.arange(52)))
def ev(K,base=2+4*x@x.T):
 es=[]
 for i,j in folds:
  p=(base+K)[np.ix_(j,i)]@solve((base+K)[np.ix_(i,i)]+.00015*np.eye(len(i)),R[i],assume_a='pos'); es.append(np.mean(abs(p-R[j]),0))
 return np.array(es)
for name, g in G.items():
 g=g/np.std(g,axis=0); t=g@g.T/4; d=np.maximum(0,np.diag(t)[:,None]+np.diag(t)[None,:]-2*t); opts=[]
 for q in [.02,.1,.5,2]:
  for r in [0,.005,.03,.15,.7]:
   for L in [.25,1,4]:
    E=ev(q*t*t+r*np.exp(-d/L)); opts.append((E[1:,0].mean(),E[0],q,r,L))
 print('kernel',name,sorted(opts,key=lambda a:a[0])[:2]);
 Q=[]
 for gg in [g,c['u']]:
  FF=poly(gg,2); res=(R if gg is g else c['v1'])-FF@fit(FF,(R if gg is g else c['v1'])); Q.append(np.std(res,axis=0));
 print('std quadratic residual',name,Q[0])
for name,g in G.items():
 q=fit(poly(g,2),R); print('quad',name,q[:,0])
for k in ['v1','v2']:
 g=c['u']; print(k,'u range',g.min(0),g.max(0));
 for f in [lambda a:a*a,lambda a:np.sin(a),lambda a:np.cos(a),lambda a:np.tanh(a),lambda a:np.abs(a)]:
  F=np.c_[aff(g),f(g)]; ee=F@fit(F,c[k])-c[k]; print(k,np.sqrt(np.mean(ee**2,0)))

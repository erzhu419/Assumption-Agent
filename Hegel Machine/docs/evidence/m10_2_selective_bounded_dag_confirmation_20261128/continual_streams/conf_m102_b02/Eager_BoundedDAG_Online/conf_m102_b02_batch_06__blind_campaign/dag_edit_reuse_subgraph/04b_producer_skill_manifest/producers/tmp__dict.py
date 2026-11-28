exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
from scipy.special import erf
from scipy.linalg import solve
from sklearn.linear_model import ARDRegression, ElasticNet, OrthogonalMatchingPursuit
p=s['panel_c'];u=p['u'];v=p['v2']; forms={'logcosh':lambda x,q:x+q[0]*np.log(np.cosh(q[1]*x)),'xtanh':lambda x,q:x+q[0]*x*np.tanh(q[1]*x),'abs':lambda x,q:x+q[0]*(np.hypot(1,q[1]*x)-1),'cos':lambda x,q:x+q[0]*(1-np.cos(q[1]*x)),'rat':lambda x,q:x+q[0]*x*x/(1+q[1]*x*x)}; rs=[]
for name,fn in forms.items():
 h=lambda a,q:D(fn(a,q)); o=least_squares(lambda q:(h(u,q)@old.fit(h(u,q),v)-v).ravel(),[1,1],bounds=([.01,.01],[20,5]));print(name,o.x,np.std(o.fun));rs.append((o.cost,fn,o.x))
_,fn,q=min(rs,key=lambda r:r[0]); c=D(fn(z,q))@old.fit(D(fn(u,q)),v); c1=D(z*np.roll(z,-1,axis=1))@old.fit(D(u*np.roll(u,-1,axis=1)),p['v1'])
np.savez('/tmp/feat2.npz',z=z,b=b,c=c,c1=c1)
Y=np.c_[t['y'],t['traj'][:,-1]]; dic={}; groups=[z,b,c1,c]; labels=['z','b','p','c'];
def add(name,a):
 for j,col in enumerate(a.T):dic[name+str(j)]=col
for label,g in zip(labels,groups):add(label,g)
for i,g in enumerate(groups):
 for j,h in enumerate(groups[:i+1]):
  label=labels[i]+labels[j]
  add(label+'*',(g[:,:,None]*h[:,None,:]).reshape(len(z),-1))
  for k in [.5,1,2]:
   add(label+'+'+str(k),(g[:,:,None]+k*h[:,None,:]).reshape(len(z),-1));add(label+'-'+str(k),(g[:,:,None]-k*h[:,None,:]).reshape(len(z),-1))
for k,x in list(dic.items()):
 std=x.std();
 if std<1e-8:continue
 for l,fn in [('sin',lambda a:np.sin(1.5*a)),('tanh',np.tanh),('asin',np.arcsinh),('rat',lambda a:a/(1+.5*a*a)),('sq',np.square),('abs',abs),('cube',lambda a:a**3),('soft',lambda a:np.maximum(abs(a)-.45,0)**1.5*np.sign(a))]:dic[l+k]=fn(x/std*.65)
X=np.column_stack(list(dic.values())); keep=X.std(0)>1e-10; X=X[:,keep]; keys=np.array(list(dic))[keep];
# Rank and prune duplicate columns before sparse feature selection.
X=(X-X[:n].mean(0))/X[:n].std(0); _,uq=np.unique(np.round(X[:n]*np.sign(X[0]),5).T,axis=0,return_index=True);X=X[:,uq];keys=keys[uq]; print('dict',X.shape)
for head in range(2):
 for i,j in sl[:2]:
  y=Y[i,head]; order=np.argsort(abs(X[i].T@(y-y.mean()))/(X[i]**2).sum(0)**.5)[::-1]; print('top',head,[(keys[k],round(np.corrcoef(X[i,k],y)[0,1],3)) for k in order[:15]])
  for a in [.002,.005,.01,.02,.04,.07]:
   m=ElasticNet(alpha=a,l1_ratio=.96,max_iter=5000,tol=1e-5).fit(X[i],y); inds=np.argsort(abs(m.coef_))[-12:][::-1]; print(head,a,np.mean(abs(m.predict(X[j])-Y[j,head])),np.mean(abs(m.predict(X[i])-y)),[(keys[k],round(m.coef_[k],3)) for k in inds if m.coef_[k]])

from probe import *
from itertools import combinations, product
from sklearn.linear_model import OrthogonalMatchingPursuit, ElasticNet, Ridge
from sklearn.model_selection import KFold

def features(s,x):
 p=s['panel_a'];u,v=p['u'],p['v'];a=np.stack((np.ones_like(u),u),axis=-1);mask=np.max(abs(v),axis=1)<3
 c=np.stack([fit(a[mask,i],v[mask,i]) for i in range(4)]);z=(x-c[:,0])/c[:,1]
 p=s['panel_b'];B=fit(aff(p['u']),p['v'])
 p=s['panel_c'];u=p['u'];edges=lambda w:w*np.roll(w,-1,axis=1);C=fit(aff(edges(u)),p['v1']);cos=lambda w:np.cos(2.4*w);D=fit(aff(cos(u)),p['v2'])
 f=lambda w:aff(edges(w))@C;g=lambda w:aff(cos(w))@D;b=lambda w:aff(w)@B
 if __name__=='__main__':print('v2 cos coefficients',D.round(4))
 V={'z':z,'x':x,'b':b(z),'f':f(z),'g':g(z),'fb':f(b(z)),'gb':g(b(z)),'bf':b(f(z)),'bg':b(g(z)),'gf':g(f(z)),'fg':f(g(z)),'fx':f(x),'gx':g(x)}
 return V

def dictionary(V,keys):
 W={};N=len(next(iter(V.values())))
 def add(name,a):
  a=np.asarray(a).reshape(N,-1)
  for j,col in enumerate(a.T): W[name+str(j)]=col
 for k in keys:
  a=V[k];add(k,a);add(k+'cube',a**3);add(k+'tanh',np.tanh(a));add(k+'sin',np.sin(2.4*a));add(k+'cos',np.cos(2.4*a))
 for i,k in enumerate(keys):
  a=V[k]
  for l in keys[i:]:
   b=V[l];P=a[:,:,None]*b[:,None,:];add(k+'*'+l,P.reshape(N,-1));add(k+'dot'+l,np.sum(a*b,axis=1));add(k+'roll'+l,np.sum(a*np.roll(b,-1,axis=1),axis=1))
 return W

if __name__=='__main__':
 V=features(s,tr['x']);h=times(t,tr['traj']);Y=np.c_[tr['y'],fit(h,tr['traj'].T).T];S=np.arange(52);T=np.arange(52,72)
 def err(p):
  r=p-Y[T];return np.r_[np.mean(abs(r),axis=0),np.mean(abs(r[:,1:]@h.T))].round(4)
 for keys in [('z','f','g'),('z','b','fb','gb'),('z','b','f','g','fb','gb'),tuple(V)]:
  print('KEYS',keys);W=dictionary(V,keys);ns=np.array(list(W));A=np.column_stack(list(W.values()));As=(A-A[S].mean(0))/np.maximum(A[S].std(0),1e-9)
  for k in [5,10,16,24,35]:
   model=OrthogonalMatchingPursuit(n_nonzero_coefs=k).fit(As[S],Y[S]);print('OMP',k,err(model.predict(As[T])),[ns[np.argsort(abs(w))[-5:]].tolist() for w in model.coef_] if k==10 else '')
  for alpha in [.0001,.001,.004,.012,.035]:
   p=[]
   for y in Y.T:
    model=ElasticNet(alpha=alpha*y[S].std(),l1_ratio=.9,max_iter=5000,tol=1e-5).fit(As[S],y[S]);p.append(model.predict(As[T]))
   print('EN',alpha,err(np.column_stack(p)))
 for keys in [('z','f','g'),('z','fb','gb'),('z','gx','fx'),('z','g','fb'),('z','f','gb'),('z','gf','fg'),('z','g'),('z','fb'),('z','f')]:
  a=np.column_stack([V[k] for k in keys]);f,g=[V[k] for k in keys[-2:]];z=V['z'];
  for name,d in [('linear',aff(a)),('basic2',aff(a,quad(z),f*g,f*np.roll(g,1,axis=1),f*np.roll(g,-1,axis=1))),('full2',aff(a,quad(a)))]:
   model=Ridge(alpha=.001,fit_intercept=False).fit(d[S],Y[S]);print(keys,name,err(model.predict(d[T])))

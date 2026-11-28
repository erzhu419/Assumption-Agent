from graphprobe import *
from itertools import combinations_with_replacement

def features(s,x):
 ai,af,B,C,D=ops(s); z=ai(x); c,d=C(z),D(z); b=B(z)
 F={};
 def add(n,v):
  v=np.asarray(v); v=v[:,None] if v.ndim==1 else v
  for j,a in enumerate(v.T): F[n+(str(j) if v.shape[1]>1 else '')]=a
 for n,u in [('z',z),('b',b),('c',c),('d',d),('CB',C(b)),('DB',D(b))]:
  add(n,u)
  for q,f in [('sq',lambda a:a*a),('cube',lambda a:a**3),('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('rat',lambda a:a/(1+a*a)),('abs',abs)]:add(f'{q}({n})',f(u))
 for n,u in [('cc',c),('dd',d),('zz',z),('bb',b)]:
  for i,j in combinations_with_replacement(range(4),2):
   if i!=j:add(n+str(i)+str(j),u[:,i]*u[:,j])
 for m,u,n,v in [('c',c,'d',d),('z',z,'c',c),('z',z,'d',d),('b',b,'c',c),('b',b,'d',d)]:
  add(m+n+'dot',np.sum(u*v,axis=1)); add(m+n+'dist',np.sum((u-v)**2,axis=1));add(m+n+'cosim',np.sum(u*v,axis=1)/np.maximum(1e-8,np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)))
  add('sum('+m+n+')',u+v);add('diff('+m+n+')',u-v)
  for i in range(4):
   for j in range(4):
    a,e=u[:,i],v[:,j]; q=m+str(i)+n+str(j)
    for name,w in [('p',a*e),('r',a/(1+e*e)),('R',e/(1+a*a)),('T',np.tanh(a+e)),('S',np.tanh(a-e)),('D',(a-e)**2)]:add(name+q,w)
 for n,u in [('c',c),('d',d),('z',z),('b',b)]:
  add(n+'sum',u.sum(1));add(n+'norm',(u*u).sum(1));add(n+'prod',u.prod(1))
  for i,j,k in combinations_with_replacement(range(4),3):add(n+str(i)+str(j)+str(k),u[:,i]*u[:,j]*u[:,k])
 return np.column_stack(list(F.values())),np.array(list(F))

def greedy(X,Y,tr,te,count):
 mu=X[tr].mean(0); X=X-mu; sc=np.linalg.norm(X[tr],axis=0); X=X/np.maximum(sc,1e-10); Ym=Y[tr].mean(); y=Y[tr]-Ym
 E=X[tr].copy(); selected=[]; vals=[]; pred=[]
 for k in range(count):
  score=abs(y@E)/np.maximum(np.linalg.norm(E,axis=0),1e-8); score[selected]=-1; j=score.argmax(); selected.append(j)
  q=E[:,j]/np.linalg.norm(E[:,j]); E-=np.outer(q,q@E); y-=q*(q@y)
  w=fit(X[tr][:,selected],Y[tr]-Ym); pred.append(X[te][:,selected]@w+Ym); vals.append(np.sqrt(np.mean(y*y)))
 return selected, np.array(pred), vals
if __name__=='__main__':
 T=S['train']; h=T['traj']; t=S['eval_times']; X,N=features(S,T['x']); r=np.array([.4,1.8,5.2]); V=-np.expm1(-t[:,None]*r); A=fit(V,h.T).T; print('amp ranges',A.min(0),A.max(0),'amp correlation',np.corrcoef(A.T)); print('time reconstruction',np.std(h-A@V.T));
 folds=[(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(72),te),te) for te in np.array_split(np.random.default_rng(123).permutation(72),5)];
 for key,y in zip(['y','terminal','slow','medium','quick','initial slope'],[T['y'],h[:,-1],*A.T,A@r]):
  e=[]
  for i,(tr,te) in enumerate(folds):
   k,p,v=greedy(X,y,tr,te,13);e.append(np.mean(abs(p-y[te]),axis=1));
   if i<2: print(key,i,list(zip(N[k],np.round(v,4))),np.round(e[-1],4))
  print('avg/max',key,np.round(np.mean(e,axis=0),4),np.round(np.max(e,axis=0),4))

from f import *
from itertools import combinations, product
P=s['panel_c'];u=P['u']; ix=[(0,1),(0,3),(1,2),(2,3)]
def Q(a): return np.column_stack([np.ones(len(a))]+[a[:,i]*a[:,j] for i,j in ix])
c=ls(Q(u),P['v1']);d=ls(A(np.cos(2.4*u)),P['v2'])
B=lambda p:A(p)@b; C=lambda p:Q(p)@c; D=lambda p:A(np.cos(2.4*p))@d
Y=np.c_[y,T]; n=52

def metrics(F):
 U,S,V=np.linalg.svd(F[:n],full_matrices=False); m=S>1e-9*S[0]; U,S,V=U[:,m],S[m],V[m]; h=(U*U).sum(1); w=U.T@Y[:n]; E=U@w-Y[:n]; H=F[n:]@(V.T@(w/S[:,None]))-Y[n:];
 a,l,v=[np.abs(g).mean(0) for g in (E,E/np.maximum(1-h,1e-10)[:,None],H)];return np.array([[p[0],p[1:].mean()] for p in (a,l,v)])

if __name__=='__main__':
 print('D',d);print('summary',tr['summary']);
 ops={'B':B,'C':C,'D':D,'I':lambda p:p,'J':lambda p:np.linalg.solve(b[1:].T,(p-b[0]).T).T}; layers={'q':q}
 for k in range(1,4):
  for word in product('BCD' if k>1 else 'BCDJ',repeat=k):
   nm=''.join(word)+'q';layers[nm]=ops[word[0]](layers[nm[1:]])
  if k==1: continue
  if k==2:
   for a in 'BCD': layers[a+'Jq']=ops[a](zi)
 res=[]
 def add(name,F):
  M=metrics(F);res.append((name,M))
 for k,p in layers.items():
  add(k,A(p));add(k+' poly2',poly(p,2));add(k+' structured Q',Q(p));
  for a in (.22,1.25):
   for lab,r in [('cos',np.cos(a*p)),('sin',np.sin(a*p)),('tanh',np.tanh(a*p)),('rdiv',p/(1+a*p*p)),('rdiv2',p*p/(1+a*p*p)),('rnorm',p/(1+a*np.sum(p*p,axis=1,keepdims=True)))]:
    add((k,a,lab),A(r));add((k,a,lab,'plus'),A(np.c_[p,r]));
 for (k,p),(j,r) in combinations(layers.items(),2):
  if min(len(k),len(j))>3:continue
  f=A(np.c_[p,r]);add((k,j,'sum'),f);add((k,j,'diag'),np.c_[f,p*r]);add((k,j,'outer'),np.c_[f,(p[:,:,None]*r[:,None,:]).reshape(-1,16)]);
  for a in (.22,1.25):
   for name,h in [('p/r',p/(1+a*r*r)),('r/p',r/(1+a*p*p)),('p/rnorm',p/(1+a*np.sum(r*r,axis=1,keepdims=True))),('r/pnorm',r/(1+a*np.sum(p*p,axis=1,keepdims=True))),('sinadd',np.sin(a*(p+r))),('cossub',np.cos(a*(p-r))),('cosadd',np.cos(a*(p+r))),('tanhadd',np.tanh(a*(p+r)))]:
    add((k,j,name,a),np.c_[f,h]);
 for j in [0,1]:
  for name,m in sorted(res,key=lambda v:v[1][1,j])[:12]:print('best',j,name,np.round(m.T,4).tolist())

exec(open('/tmp/grammar.py').read().split('f=PolynomialFeatures(5)')[0])
from scipy.optimize import least_squares
p=s['panel_c']; u=p['u']; print('panel norms',np.max(abs(u),axis=0))
def fitmap(key, kind, ids):
 a=u[ids]; b=p[key][ids]; U=lambda X:np.c_[np.ones(len(X)),X];
 def feat(X,q):
  Z=X*q[:4]+q[4:]; h={'tanh':lambda:np.tanh(Z),'sin':lambda:np.sin(Z),'xtanh':lambda:X*np.tanh(Z),'sat2':lambda:X**2/(1+Z**2),'cos':lambda:np.cos(Z),'logcosh':lambda:np.logaddexp(Z,-Z)}[kind]();return np.c_[U(X),h]
 def prof(q):
  F=feat(a,q); w=np.linalg.lstsq(F,b,rcond=None)[0]; return F@w-b,w
 best=None
 for k in ([1.5,2.5,3.5] if key=='v1' else [.4,.8,1.5]):
  sol=least_squares(lambda q:prof(q)[0].ravel(),np.r_[np.full(4,k),np.zeros(4)],bounds=(np.r_[np.full(4,.03),np.full(4,-1.)],np.r_[np.full(4,5.),np.ones(4)]),max_nfev=120)
  if best is None or sol.cost<best.cost:best=sol
 q=best.x; e,w=prof(q); return q,w,np.sqrt(np.mean(e**2)),lambda X:feat(X,q)@w
maps={}
for key,kinds in [('v1',['tanh','sin']),('v2',['xtanh','sat2','cos','logcosh'])]:
 for k in kinds:
  q,W,r,g=fitmap(key,k,np.arange(24)); loo=[]
  for v in range(24):
   _,_,_,h=fitmap(key,k,np.delete(np.arange(24),v));loo.append(h(u[v:v+1])[0]-p[key][v]);
  print(key,k,q.round(4),r,np.sqrt(np.mean(np.asarray(loo)**2)), '\n',W.round(4)); maps[key,k]=g
P=lambda X,d:PolynomialFeatures(d).fit_transform(X)
rows=[]
for name,XX in [('raw',x),('inverse',xf),('direct',x*A[0][1:].diagonal()+A[0][0])]:
 G=np.c_[np.ones(len(XX)),XX]; C=G@A[2]; Z=maps['v1','tanh'](XX); V=maps['v2','xtanh'](XX); R=G@A[1]; S=C/np.sqrt(1+.09*C*C); 
 for label,F in [('C',P(S,3)),('Z',P(Z,3)),('V',P(V,3)),('R',P(R,3)),('ZV',P(np.c_[Z,V],2)),('Z+V',np.c_[P(Z,3),V]),('S+ZV',np.c_[P(S,3),.5*P(np.c_[Z,V],2)]),('S+V',np.c_[P(S,3),.5*P(V,2)]),('X+ZV',np.c_[P(XX,2),P(np.c_[Z,V],2)]),('X+Z+V',np.c_[P(XX,2),Z,V]),('R+ZV',np.c_[P(R,2),P(np.c_[Z,V],2)]),('all',np.c_[P(XX,2),P(np.c_[Z,V,R],2)])]:
  for reg in [.001,.006,.012,.025,.05]:
   T=scores(F,reg);rows.append((T[2].sum(),name,label,reg,T))
for _,nm,lab,r,T in sorted(rows,key=lambda a:a[0])[:20]:print(nm,lab,r,T.round(5).tolist())

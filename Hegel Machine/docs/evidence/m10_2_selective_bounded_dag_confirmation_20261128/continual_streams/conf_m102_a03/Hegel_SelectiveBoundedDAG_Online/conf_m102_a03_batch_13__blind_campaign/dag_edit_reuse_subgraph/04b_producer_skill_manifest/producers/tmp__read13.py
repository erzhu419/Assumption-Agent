from struct13 import *
u,v=s['panel_b']['u'],s['panel_b']['v']
def bf(a,h): return np.c_[aff(a),np.sin(h*a),np.cos(h*a)]
def res(h):
 A=bf(u,h[0]);return (A@fit(A,v)-v).ravel()
o=min([least_squares(res,[h],bounds=(.1,8)) for h in (1.,2.,3.,4.,5.,6.,7.)],key=lambda a:a.cost); h=o.x[0]; bc=fit(bf(u,h),v); b=bf(z,h)@bc
if __name__=='__main__':
 for d in (1,2,3,4):
  F=poly(b,d); print('B poly',d,[(l,scores(F,y,l)) for l in (.00001,.001,.01,.1,1)])
 print('state coeff',fit(aff(b),T).round(4)); print('B coef',bc.round(5)); print('Y coefficients',fit(poly(b,2),y).round(4)); print('linear y residual',np.c_[b,y, y-aff(b)@fit(aff(b),y)][:25].round(3))
 def ps(D,labels):
  for k in (1,2,3,4,5,6,8,10,14,20):
   es=[]
   for a,j in splits:
    A=aff(b); ids=pursuit(A[a],D[a],y[a,None],k,()); F=np.c_[A,D[:,ids]]; es.append(np.mean(abs(F[j]@fit(F[a],y[a])-y[j])))
   print(k,np.round([es[0],np.median(es),max(es)],4),[labels[i] for i in ids[:8]])
 D=[]; labels=[]
 def add(F,L):D.extend(np.asarray(F).reshape(n,-1).T);labels.extend(L)
 for k in (2,3,4):add(poly(b,k,k),['b'+str(p) for p in it.combinations_with_replacement(range(4),k)])
 for x0,name in ((b,'b'),(z,'z')):
  for c in (.5,1.,1.5,2.,2.6,3.,4.):
   for f,fn in ((np.sin,'sin'),(np.cos,'cos'),(np.tanh,'tanh'),(lambda p:p/(1+p*p),'rat')):add(f(c*x0),[(name,fn,c,j) for j in range(4)])
 for a0,j in it.combinations(range(4),2):
  for sign in (-1,1):
   g=b[:,a0]+sign*b[:,j]
   for c in (.5,1.,2.,3.,4.):
    for f,fn in ((np.sin,'sin'),(np.cos,'cos'),(np.tanh,'tanh'),(lambda p:p/(1+p*p),'rat')):add(f(c*g),[(a0,j,sign,fn,c)])
 for name,F in [('all',np.c_[aff(b),poly(b,3,2),z]),('mixed',np.c_[poly(b,2),poly(z,2,1),(b[:,:,None]*z[:,None,:]).reshape(n,-1)]),('trig',np.c_[aff(b),bf(z,h),poly(b,2,2)])]: print(name,[(l,scores(F,y,l)) for l in (.001,.01,.1,1)])
 ps(np.array(D).T,labels)

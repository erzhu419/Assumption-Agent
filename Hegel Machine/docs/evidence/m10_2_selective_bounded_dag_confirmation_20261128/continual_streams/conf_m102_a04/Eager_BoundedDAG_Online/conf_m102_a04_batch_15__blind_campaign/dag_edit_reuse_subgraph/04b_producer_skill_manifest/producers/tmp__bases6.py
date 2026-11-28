import pickle,numpy as np
from itertools import combinations,combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=4,suppress=True)
def L(f,y):return np.linalg.lstsq(f,y,rcond=1e-9)[0]
def evalf(f,y):
 b=L(f,y); e=y-f@b; h=(np.linalg.svd(f,full_matrices=False)[0]**2).sum(1);return np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/np.maximum(1-h,1e-8)[:,None])**2)),b
for key,ykey in [('panel_a','v'),('panel_b','v'),('panel_c','v2')]:
 x=s[key]['u'];y=s[key][ykey]; n,d=x.shape; F=np.c_[np.ones(n),x]; good=np.ones(n,bool)
 if key=='panel_a':
  q=least_squares(lambda q:(F@q.reshape(5,4)-y).ravel(),L(F,y).ravel(),loss='cauchy',f_scale=.18).x.reshape(5,4);good=np.max(abs(F@q-y),axis=1)<2;x=x[good];y=y[good];F=F[good]; n=len(x)
 print('\n',key, 'linear',evalf(F,y)[:2]);print('linear coef',L(F,y))
 for degree in [2,3]:
  terms=list(combinations_with_replacement(range(d),degree)); D=np.column_stack([np.prod(x[:,t],axis=1) for t in terms]); ff=np.c_[F,D]; m,cv,b=evalf(ff,y); print('add degree',degree,m,cv)
  for j in range(d): print('largest',j,[(terms[i],round(b[len(F.T)+i,j],4)) for i in np.argsort(abs(b[len(F.T):,j]))[-6:][::-1]])
  if degree==2 and key=='panel_a': F=ff
 B=np.c_[np.ones(n),x];roll=lambda k:np.roll(x,k,axis=1)
 if key=='panel_a':
  for j in range(4):
   a=x[:,j];t=x[:,(j+1)%4];p=x[:,(j-1)%4];yy=y[:,j,None];
   families={'cyclic':np.c_[a*t,a*p,t*p,a*a,t*t,p*p,a**3,t**3,p**3], 'mixodd':np.c_[a*a*t,a*t*t,a*a*p,a*p*p,a*t*p], 'hinges':np.c_[np.maximum(a,0),np.maximum(t,0),np.maximum(p,0),a*abs(a),t*abs(t),p*abs(p)]}
   for name,g in families.items(): print(j,name,evalf(np.c_[B,g],yy)[:2],L(np.c_[B,g],yy).ravel())
 if key!='panel_a':
  for j in range(4):
   yy=y[:,j,None]; scores=[]
   for i,k in combinations(range(4),2):
    a,t=x[:,i],x[:,k];
    for nm,r in [('prod',a*t),('sum',a+t),('diff',a-t),('a',a),('t',t)]:
     for typ,fun in [('sin',lambda q:np.sin(q[0]*r)),('cos',lambda q:np.cos(q[0]*r)),('sat',lambda q:r/(1+q[0]*r*r)),('even',lambda q:r*r/(1+q[0]*r*r)),('tanh',lambda q:np.tanh(q[0]*r)),('sinprod',lambda q:np.sin(q[0]*a)*t),('damp',lambda q:a/(1+q[0]*t*t)),('damp2',lambda q:a*t/(1+q[0]*t*t)),('soft',lambda q:r/(1+q[0]*abs(r)))]:
      def design(q):return np.c_[B,fun(q)]
      def res(q):
       ff=design(q);return (ff@L(ff,yy)-yy).ravel()
      opts=[least_squares(res,[q],bounds=(.003,9)).x for q in [.5,2.1,4.]];q=min(opts,key=lambda q:np.linalg.norm(res(q))); m,cv,_=evalf(design(q),yy);scores.append((cv,m,(i,k),nm,typ,q[0]))
   print(j,sorted(scores)[:7])

exec(open('/tmp/grammar.py').read().split('rows=[]')[0])
from scipy.optimize import least_squares
N=52; it=np.arange(N); va=np.arange(N,72)
np.set_printoptions(precision=4,suppress=True)
def polynomial(z,d):return np.column_stack([np.prod(z[:,j],axis=1) for k in range(d+1) for j in combinations_with_replacement(range(4),k)])
def rr(F,j):
 X=F[it]; P=np.linalg.pinv(X); r=Y[it,j]-X@(P@Y[it,j]); lev=np.sum(X*P.T,1); return np.mean(abs(r)),np.mean(abs(r)/(1-lev)),np.mean(abs(F[va]@(P@Y[it,j])-Y[va,j]))
rows=[]
for key in ['x','a','a0','a1','x0','x1','ba','ba1','Ba','Ba1']:
 z=trans[key][:72]; p=[polynomial(z,d) for d in range(4)]; v=[('norm',np.sum(z*z,1)),('sum',np.sum(z,1)**2),('l1',np.sum(abs(z),1)),('norm4',np.sum(z**4,1))]; v.extend((str(i),z[:,i]**2) for i in range(4)); v.extend((str((i,j,sign)),(z[:,i]+sign*z[:,j])**2) for i,j in combinations(range(4),2) for sign in [-1,1]); v.extend((str((i,j)),z[:,i]**2+z[:,j]**2) for i,j in combinations(range(4),2))
 for j in range(3):
  for d in [1,2,3]:
   for name,w in v:
    for q in [.11,.22,.44,.8,1.2]:
     den=(1+q*w)[:,None]
     for mode in [0,1]:
      F=p[d]/den if not mode else np.column_stack((p[1],p[d]/den)); rows.append((*rr(F,j),j,key,d,name,q,mode))
for j in range(3):
 r=[r for r in rows if r[3]==j];print('j',j,'fixed best',*sorted(r,key=lambda x:x[1])[:8],sep='\n')
rows=[]
for key in ['x','a','a1','ba','ba1','Ba','Ba1']:
 z=trans[key][:72]; p=[polynomial(z,d) for d in range(4)]
 for d in [2,3]:
  for mode in ['rad','diag','center','ridge','full']:
   def den(q):
    if mode=='rad':return 1+q[0]*np.sum(z*z,1)
    if mode=='diag':return 1+(z*z)@q
    if mode=='center':return 1+((z-q[4:])**2)@q[:4]
    if mode=='ridge':return 1+(z@q[:4]+q[4])**2
    return 1+np.sum((z@q.reshape(4,4))**2,1)
   q0={'rad':[.22],'diag':[.22]*4,'center':[.22]*4+[0]*4,'ridge':[.46]*4+[0],'full':(.46*np.eye(4)).ravel()}[mode]
   for cols in [[0,1,2],[0],[1],[2]]:
    yy=Y[it][:,cols]; std=yy.std(0)
    def ob(q):
     F=p[d][it]/den(q)[it,None];return ((F@fit(F,yy)-yy)/std).ravel()
    lb=np.full(len(q0),-3.);ub=np.full(len(q0),3.);
    if mode in ('rad','diag','center'):lb[:min(4,len(q0))]=0
    o=least_squares(ob,q0,bounds=(lb,ub),max_nfev=75); F=p[d]/den(o.x)[:,None];
    for j in cols:rows.append((*rr(F,j),j,key,d,mode,cols,o.x))
for j in range(3):
 r=[r for r in rows if r[3]==j];print('j',j,'profile best',*sorted(r,key=lambda x:x[1]*(1+len(x[-1])/35))[:8],sep='\n');print('tail diagnostic',*sorted(r,key=lambda x:x[2])[:3],sep='\n')

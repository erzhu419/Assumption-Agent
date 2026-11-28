exec(open('/tmp/explore.py').read().split('for key')[0]); from itertools import combinations,product
A=s['panel_a']; keep=np.linalg.norm(A['u']-A['v'],axis=1)<2; g,o=np.array([np.polyfit(A['u'][keep,j],A['v'][keep,j],1) for j in range(4)]).T; z=(x-o)/g
B=s['panel_b']; C=s['panel_c']; b=np.linalg.lstsq(pol(B['u'],1),B['v'],rcond=None)[0]; p=C['u'];
r=least_squares(lambda a:(T[:n]-T[:n]@np.linalg.pinv(-np.expm1(-np.exp(a)[:,None]*t))@(-np.expm1(-np.exp(a)[:,None]*t))).ravel(),[-1,.6]); q=-np.expm1(-t[:,None]*np.exp(r.x)); Y=np.c_[y,T@np.linalg.pinv(q.T)]
def aff(x):return pol(x,1)
def lib(x):
 d={}; a=list(x.T)
 def add(k,v):d[k]=v
 for k in range(2,5):
  for ids in combinations_with_replacement(range(4),k):add(str(ids),np.prod(x[:,ids],axis=1))
 for i,j in combinations(range(4),2):
  u,v=a[i],a[j]; add(f'rat{i}{j}',u*v/(1+.5*(u*u+v*v)))
 w={str(i):u for i,u in enumerate(a)}
 for i,j in combinations(range(4),2):
  for sgn in (1,-1):w[f'{i},{sgn*j:+}']=a[i]+sgn*a[j]
 for key,u in w.items():
  for fr in (.5,1,1.5,2,2.5,3):
   for nm,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('exp',lambda x:np.exp(-x*x)),('sat',lambda x:x/(1+abs(x))),('atan',np.arctan)]:add(f'{nm}{fr}:{key}',fn(fr*u))
 return list(d),np.column_stack(list(d.values()))
def omp(x,Y,fit,val,k):
 F=aff(x); names,D=lib(x); R=D[fit]-F[fit]@np.linalg.lstsq(F[fit],D[fit],rcond=None)[0]; sd=np.linalg.norm(R,axis=0); R/=np.maximum(sd,1e-10); out=[]
 for i in range(Y.shape[1]):
  A=F.copy(); rr=Y[fit,i]-A[fit]@np.linalg.lstsq(A[fit],Y[fit,i],rcond=None)[0]; use=[]; scores=[]
  for it in range(k):
   j=np.argmax(abs(R.T@rr)); use.append(names[j]); A=np.c_[A,D[:,j]]; w=np.linalg.lstsq(A[fit],Y[fit,i],rcond=None)[0];rr=Y[fit,i]-A[fit]@w; scores.append((round(float(rms(rr)),5),round(float(mae(Y[val,i]-A[val]@w)),5)))
  out.append((use,scores))
 return out
for nm,v in [('c1',C['v1']),('c2',C['v2'])]:
 print(nm)
 for row in omp(p,v,np.arange(19),np.arange(19,24),8):print(row)
for nm,v in [('z',z),('b',aff(z)@b),('bi',(z-b[0])@np.linalg.inv(b[1:]))]:
 print(nm)
 for row in omp(v,Y,np.arange(52),np.arange(52,72),15):print(row)

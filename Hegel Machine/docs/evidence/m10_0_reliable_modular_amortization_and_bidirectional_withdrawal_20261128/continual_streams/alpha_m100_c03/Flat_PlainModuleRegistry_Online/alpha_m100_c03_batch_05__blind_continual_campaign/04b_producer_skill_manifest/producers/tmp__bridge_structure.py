import numpy as np, pickle
from itertools import combinations
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; X=D['true_channel_loads']; Y=D['bridge_raw']; good=np.max(abs(Y),axis=1)<5
axis=np.flatnonzero(good & (np.count_nonzero(X,axis=1)==1)); mix=np.flatnonzero(good & (np.count_nonzero(X,axis=1)!=1)); splits=[('mix',v) for v in np.array_split(np.random.default_rng(49).permutation(mix),4)]+[('axis',axis),('extreme',mix[np.max(abs(X[mix]),axis=1)>1.48])]
for i in range(4):
 rows=axis[X[axis,i]!=0]; print('axis input',i,'x=',X[rows,i], 'y=',Y[rows].round(6))

def design(x,j,kind,p):
 s=x[:,j]; t=x[:,(j+2)%4]; a=p[0]; b=p[-1]; A=[np.ones(len(x)),*x.T,np.sin(a*s),t*t]; names=['1',*map(str,range(4)),'sin(own)','cross^2']; cols={
 'sin': (np.sin(b*t),'sin(cross)'), 'cos': (np.cos(b*t)-1,'cos(cross)'), 'tanh': (np.tanh(b*t),'tanh(cross)'), 'atan': (np.arctan(b*t),'atan(cross)'), 'soft': (t/np.sqrt(1+b*t*t),'soft(cross)'), 'rat': (t/(1+b*t*t),'rat(cross)'), 'cub': (t**3,'cross^3'), 'own2':(s*s,'own^2'), 'owncos':(np.cos(a*s)-1,'cos(own)'), 'own3':(s**3,'own^3'), 'prod':(s*t,'own*cross'), 'prodsin':(np.sin(a*s)*t,'sin(own)*cross'), 'sinprod':(np.sin(b*s*t),'sin(own*cross)'), 'sum':(np.sin(b*(s+t)),'sin(own+cross)'), 'dif':(np.sin(b*(s-t)),'sin(own-cross)'), 'quadall': (np.column_stack([x[:,i]*x[:,k] for i in range(4) for k in range(i,4) if not i==k==(j+2)%4]),'quad'), 'sinall':(np.sin(b*x[:,[i for i in range(4) if i!=j]]),'sinall')}
 for k in kind:
  col,name=cols[k]; A.append(col); names.append(name)
 return np.column_stack(A),names

def fit(j,kind,t):
 def params(p):
  B=design(X[t],j,kind,p)[0]; beta=np.linalg.lstsq(B,Y[t,j],rcond=None)[0]; return B@beta-Y[t,j],beta
 lo=[1.8,.08]; hi=[3.2,6.]; starts=[[2.5,b] for b in (.7,1.5,2.5,4.)] if set(kind)&{'sin','cos','tanh','atan','soft','rat','sinall','sinprod','sum','dif'} else [[2.5,2.5]]
 obj=min((least_squares(lambda p:params(p)[0],p,bounds=(lo,hi),max_nfev=60) for p in starts),key=lambda r:r.cost); return obj.x,params(obj.x)[1]
for kind in [(),('sin',),('tanh',),('atan',),('rat',),('cub',),('own2',),('owncos',),('prod',),('sinprod',),('sum',),('dif',),('sin','own2'),('sin','owncos'),('sin','prod'),('sin','prodsin'),('sin','cos','own2'),('sin','quadall'),('sinall','quadall')]:
 errors={}; fits=[]; coef=[]
 for j in range(4):
  p,c=fit(j,kind,good); yh=design(X,j,kind,p)[0]@c; fits.append(np.sqrt(np.mean((Y[good,j]-yh[good])**2))); coef.append([p.round(4).tolist(),c.round(5).tolist()]);
  for name,v in splits:
   t=good.copy();t[v]=False; q,b=fit(j,kind,t); e=design(X[v],j,kind,q)[0]@b-Y[v,j]; a=errors.setdefault(name,np.zeros(4));a[j]=max(a[j],np.sqrt(np.mean(e*e)))
 print(kind, 'fit',np.round(fits,6),'CV',{n:v.round(6).tolist() for n,v in errors.items()},'coef',coef,flush=True)

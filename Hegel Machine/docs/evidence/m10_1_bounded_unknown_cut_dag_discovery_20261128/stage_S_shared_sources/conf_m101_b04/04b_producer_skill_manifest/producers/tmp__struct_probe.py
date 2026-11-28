import pickle, numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=[s['train'][k] for k in ('x','y','traj')]; t=s['eval_times']; n=len(x)
ls=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
def poly(v,d): return np.column_stack([np.ones(len(v))]+[np.prod(v[:,j],1) for k in range(1,d+1) for j in combinations_with_replacement(range(4),k)])
def kern(u,v,M,h):
 d=(u[:,None]-v[None,:])@M
 return np.exp(-h*np.sum(d*d,2))
def fitkernel(v,u,Y,M,h,pen):
 P=poly(v,2); K=kern(v,v,M,h)+pen*np.eye(len(v)); C=np.linalg.solve(np.block([[K,P],[P.T,np.zeros((15,15))]]),np.concatenate((Y,np.zeros((15,)+Y.shape[1:])))); return np.c_[kern(u,v,M,h),poly(u,2)]@C
p=s['panel_a']; d=(p['v']-np.median(p['v']-p['u'],0)); scale=np.array([np.median((d[:,j]-d[i,j])[(abs(p['u'][:,j]-p['u'][i,j])>.3)]/(p['u'][:,j]-p['u'][i,j])[(abs(p['u'][:,j]-p['u'][i,j])>.3)]) for j in range(4) for i in [0]]); B=ls(poly(s['panel_b']['u'],1),s['panel_b']['v'])[1:]; print('cal scale',scale,'B',B)
splits=[np.arange(52,72),np.argsort(np.linalg.norm(x,axis=1))[-18:]]+[ix for j in range(4) for order in [np.argsort(x[:,j])] for ix in (order[:18],order[-18:])]; splits+=list(np.array_split(np.random.default_rng(903).permutation(n),4))
results={}
for va in splits:
 tr=np.setdiff1d(np.arange(n),va); Z=z[tr].T; P=poly(x,2)
 def add(key,p,q): results.setdefault(key,[]).append([np.mean(abs(p-y[va])),np.mean(abs(q-z[va])),np.mean(abs(q-z[va]),0).max()])
 def fitbasis(mode):
  def bas(k):
   E=-np.expm1(-t[:,None]*k[:2]); f=t*np.exp(-k[0]*t)
   if mode=='deriv': return np.c_[E,f]
   if mode=='third': return np.c_[E,-np.expm1(-t*k[2])]
   if mode=='sine': return np.c_[E,np.sin(k[2]*t)]
   return E
  init,lo,hi=[1.5,.25],[.7,.02],[4,.65]
  if mode in ('third','sine'): init+=[3.];lo+=[2.];hi+=[8.]
  k=least_squares(lambda k:(bas(k)@ls(bas(k),Z)-Z).ravel(),init,bounds=(lo,hi)).x
  return bas(k)
 b=fitbasis('two'); yy=fitkernel(x[tr],x[va],y[tr],np.eye(4),.23,.001); zz=fitkernel(x[tr],x[va],ls(b,Z).T,np.eye(4),.23,.006)@b.T
 add('baseline',yy,zz)
 for mode in ['deriv','third','sine']:
  c=fitbasis(mode); q=fitkernel(x[tr],x[va],ls(c,Z).T,np.eye(4),.23,.006)@c.T; add('time '+mode,yy,q)
 # Estimate a single-index saturating scalar link, retaining a free affine component.
 for law,fun in [('tanh',np.tanh),('sin',np.sin),('soft',lambda w:w/np.sqrt(1+w*w))]:
  L=P[:,:5]; Q=P[:,5:]; A=P[tr]; yt=y[tr]; init=ls(A,yt); opts=[]
  for amp in [.7,1.5,3.]:
   def pred(k,rr):return L[rr]@k[:5]+k[-1]*fun(P[rr]@k[5:-1])
   o=least_squares(lambda k:np.r_[pred(k,tr)-yt,.004*k[5:-1]],np.r_[np.zeros(5),init/amp,amp],max_nfev=500)
   opts.append(o)
  add('link '+law,pred(min(opts,key=lambda o:o.cost).x,va),zz)
 # Compare raw and operator-induced kernel geometries, including smoother extrapolation.
 inv=np.diag(1/scale); I=np.eye(4); H=np.c_[I,B]; norm=lambda M: M*np.sqrt(4/np.sum(M*M))
 for lab,M in [('raw',I),('cal',norm(inv)),('B',norm(inv@B)),('join',norm(inv@H))]:
  for width,pen in [(.09,.0002),(.15,.0005),(.23,.001),(.34,.002)]:
   yp=fitkernel(x[tr],x[va],y[tr],M,width,pen)
   zp=fitkernel(x[tr],x[va],ls(b,Z).T,M,width,pen*6)@b.T
   add((lab,width),yp,zp)
for key,res in results.items():
 a=np.array(res); print(key,'fixed',np.round(a[0],5),'tails max',np.round(a[1:10].max(0),5),'random max',np.round(a[10:].max(0),5),'tail y',np.round(a[1:10,0],4))

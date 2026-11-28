import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
a=s['panel_a']; u=a['u']; v=a['v']; good=np.max(abs(v-u),1)<3;u=u[good];v=v[good]
for d in (1,2,3):
 ii=[k for j in range(1,d+1) for k in itertools.combinations_with_replacement(range(4),j)]; X=np.c_[np.ones(len(u)),*[np.prod(u[:,k],1) for k in ii]]; c=ls(X,v); print('A poly',d,'RMS',np.std(X@c-v,0));
 if d==3:
  for j in range(4):print(j,[(str(k),round(x,4)) for k,x in zip([(),*ii],c[:,j]) if abs(x)>.008])
u,v=[s['panel_b'][k] for k in ('u','v')]; i=np.arange(len(u)); fold=i%4==0
funs={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'asinh':np.arcsinh,'rational':lambda z:z/(1+z*z),'cubic':lambda z:z**3}
for key,f in funs.items():
 for kind in ('input','input+linear','output'):
  def design(p,x):
   z=f(x*p[:4]+p[4:]);return np.c_[np.ones(len(x)),z] if kind!='input+linear' else np.c_[np.ones(len(x)),x,z]
  def pred(p,mask,evalmask):
   if kind!='output':return design(p,u[evalmask])@ls(design(p,u[mask]),v[mask]);
   return f(np.c_[np.ones(sum(evalmask)),u[evalmask]]@p.reshape(5,4))
  vals=[]
  for m in (~fold,np.ones(len(u),bool)):
   p0=np.r_[np.ones(4),np.zeros(4)] if kind!='output' else ls(np.c_[np.ones(len(u)),u],v).ravel(); opt=least_squares(lambda p:(pred(p,m,m)-v[m]).ravel(),p0,max_nfev=180); predm=~m if (~m).any() else m; vals.append(np.mean(abs(pred(opt.x,m,predm)-v[predm])));p=opt.x
  print('B',key,kind,np.round(vals,5),np.round(p,3) if min(vals)<.024 else '')

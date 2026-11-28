import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def one(x):return np.column_stack([np.ones(len(x)),x])
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def cal(p):
 u,v=p['u'],p['v']; h=one(u); good=np.argsort(np.linalg.norm(h@ls(h,v)-v,axis=1))[:45]; scores=[]
 # Each output involves primarily its own coordinate and the antipodal coordinate.
 xx=[]
 for j in range(4):
  a,b=u[:,j],u[:,(j+2)%4]; y=v[:,j]; wins=[]
  for nm,f in [('tanh',np.tanh),('atan',np.arctan),('sin',np.sin),('soft',lambda x:x/np.sqrt(1+x*x)),('rat',lambda x:x/(1+abs(x))),('cub',lambda x:x**3)]:
   for knd in range(4):
    def mat(k):
     if knd==0:return one(np.column_stack([a,f(k[0]*b)]))
     if knd==1:return one(np.column_stack([f(k[0]*a),f(k[1]*b)]))
     if knd==2:return one(np.column_stack([a,b,f(k[0]*a),f(k[1]*b)]))
     return one(np.column_stack([a,b,f(k[0]*(a+k[1]*b))]))
    def r(k):
     h=mat(k)[good];return h@ls(h,y[good])-y[good]
    opt=least_squares(r,[1.25,1.25],bounds=([.035,-4] if knd==3 else [.035,.035],[6,6]),max_nfev=200)
    wins.append((np.std(opt.fun),nm,knd,opt.x,ls(mat(opt.x)[good],y[good])))
  print(j,sorted(wins,key=lambda x:x[0])[:6])
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; h=one(p['u']);
 for key,y in p.items():
  if key=='u':continue
  if k=='panel_a':cal(p);break
  if key=='v2':h=one(np.cos(2.4*p['u']))
  c=ls(h,y); print(k,key,'coefs',np.round(c,5),'resid',np.std(h@c-y));

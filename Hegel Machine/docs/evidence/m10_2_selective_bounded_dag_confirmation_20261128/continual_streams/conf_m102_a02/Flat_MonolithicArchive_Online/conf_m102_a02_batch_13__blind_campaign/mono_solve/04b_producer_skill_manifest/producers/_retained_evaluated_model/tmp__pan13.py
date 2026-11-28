import numpy as np,pickle
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda d,y:np.linalg.lstsq(d,y,rcond=None)[0]
def fit(d,y):
 r=d@ls(d,y)-y;l=np.sum(np.linalg.svd(d,full_matrices=False)[0]**2,axis=1);return np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/(1-l[:,None]))**2))
for key,out in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 x=s[key]['u']; y=s[key][out]; a=np.c_[np.ones(len(x)),x]; nxt=np.roll(x,-1,axis=1); opp=np.roll(x,-2,axis=1); prev=np.roll(x,1,axis=1)
 gs={'x':x,'square':x*x,'cube':x**3,'nextprod':x*nxt,'opprod':(x*opp)[:,:2],'sum':x+nxt,'diff':x-nxt,'oppsum':(x+opp)[:,:2],'oppdiff':(x-opp)[:,:2]}; fs={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'rational':lambda z:z/(1+z*z),'rational2':lambda z:z*z/(1+z*z),'quadadd':lambda z:z+z*z}; results=[]
 for gn,g in gs.items():
  for fn,f in fs.items():
   for mode in range(3):
    def d(p):
     z=f(p*g);return np.c_[a[:,:1],z] if mode==0 else np.c_[a,z] if mode==1 else np.c_[a,f if False else (np.sin(p*g) if fn=='cos' else np.cos(p*g)),z]
    ps=np.linspace(.15,8,60); j=np.argmin([fit(d(p),y)[0] for p in ps]); p=minimize_scalar(lambda p:fit(d(p),y)[0],bounds=(max(.03,ps[j]-.17),ps[j]+.17),method='bounded').x;r,l=fit(d(p),y);results.append((l,r,gn,fn,mode,p))
 print(key,out,sorted(results)[:15]);
 for fn,f in fs.items():
  print(fn,[(p,fit(np.c_[a,f(p*x)],y)) for p in (.22,1.25,2.15)])

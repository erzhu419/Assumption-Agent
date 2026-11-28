import numpy as np
from scipy.optimize import least_squares

def lin(f,y): return np.linalg.lstsq(f,y,rcond=1e-9)[0]
def one(z): return np.c_[np.ones(len(z)),z]

def fitc(p,verbose=False):
 u=p['u']; v=p['v1']; inds=np.arange(len(u)); models=[]
 funcs={'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'sinh':np.sinh,'asinh':np.arcsinh,'rational':lambda x:x/(1+x*x),'erf':lambda x: __import__('scipy').special.erf(x)}
 for typ,fun in funcs.items():
  for mode in ['shared','scale','shift','rotation','rotshift']:
   if mode.startswith('rot') and typ not in ['tanh','atan','sin','erf']:continue
   def feat(x,h,mode=mode,fun=fun):
    if mode=='shared':w=x*h[0]
    elif mode=='scale':w=x*h
    elif mode=='shift':w=x*h[:4]+h[4:]
    else:w=x@h[:16].reshape(4,4)+(h[16:] if mode=='rotshift' else 0)
    return np.c_[one(x),fun(w)]
   init={'shared':[1.6],'scale':[1.6]*4,'shift':[1.6]*4+[0]*4,'rotation':(1.6*np.eye(4)).ravel(),'rotshift':np.r_[(1.6*np.eye(4)).ravel(),[0]*4]}[mode]
   def train(i,init):
    def err(h):
     f=feat(u[i],h);return (f@lin(f,v[i])-v[i]).ravel()
    z=least_squares(err,init,bounds=(-5,5),max_nfev=120);return z.x,lin(feat(u[i],z.x),v[i]),np.sqrt(np.mean(z.fun**2))
   hv,b,e=train(inds,init)
   if not np.isfinite(e):continue
   cv=[]
   for j in np.array_split(np.random.default_rng(42).permutation(inds),4):
    h,q,_=train(np.setdiff1d(inds,j),init);cv.extend((feat(u[j],h)@q-v[j]).ravel())
   score=np.sqrt(np.mean(np.array(cv)**2));models.append((score,e,typ,mode,hv,b,feat));
   if verbose:print(typ,mode,'panel fit and cv',round(e,6),round(score,6),'params',hv.round(3),flush=True)
 models.sort(key=lambda z:z[0]);return [(lambda x,f=f,h=h,b=b:f(x,h)@b) for _,_,_,_,h,b,f in models[:3]]

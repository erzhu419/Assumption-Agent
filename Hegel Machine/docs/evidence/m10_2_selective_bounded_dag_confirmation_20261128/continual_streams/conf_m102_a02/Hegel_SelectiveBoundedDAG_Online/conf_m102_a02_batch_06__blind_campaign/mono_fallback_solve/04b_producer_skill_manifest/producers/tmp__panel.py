exec(open('/tmp/probe.py').read().split('np.set_printoptions')[0])
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True)
def fit(P,key):
 U,V=P['u'],P[key]; ls=np.linalg.lstsq
 funcs={'tanh':lambda a,k:np.tanh(k*a),'rat':lambda a,k:a/(1+k*a*a),'atan':lambda a,k:np.arctan(k*a),'asinh':lambda a,k:np.arcsinh(k*a),'sin':lambda a,k:np.sin(k*a),'sqrt':lambda a,k:np.sqrt(1+k*a*a),'erf':lambda a,k:__import__('scipy').special.erf(k*a),'softsign':lambda a,k:a/np.sqrt(1+k*a*a)}
 out=[]
 for n,h in funcs.items():
  for mode in ['mix','pure','shift','shiftmix','post']:
   def design(a,p):
    k,b,c=p; return np.c_[np.ones(len(a)), h(a+(c if 'shift' in mode else 0),k)+(b*a if 'mix' in mode else 0)]
   def pred(a,p,B):
    return design(a,p)@B if mode!='post' else np.column_stack([h(a@B[1:,j]+B[0,j],p[0])+p[1]*(a@B[1:,j]+B[0,j])+p[2] for j in range(4)])
   def res(p):
    F=design(U,p); return (F@ls(F,V,rcond=None)[0]-V).ravel()
   if mode=='post':
    F=np.c_[np.ones(len(U)),U]; B0=ls(F,V,rcond=None)[0]; O=least_squares(lambda c:(pred(U,c[:3],c[3:].reshape(5,4))-V).ravel(),np.r_[1.25,.22,0,B0.ravel()],max_nfev=180); p,B=O.x[:3],O.x[3:].reshape(5,4)
   else:
    O=least_squares(res,[1.25,.22,0],bounds=([.08,-4,-1],[8,4,1]),max_nfev=180);p=O.x;B=ls(design(U,p),V,rcond=None)[0]
   out.append((np.std(O.fun),n,mode,p,B,lambda a,p=p,B=B,pred=pred,design=design,mode=mode,h=h: np.c_[np.ones(len(a)),h(a+(p[2] if 'shift' in mode else 0),p[0])+(p[1]*a if 'mix' in mode else 0)]@B if mode!='post' else h(np.c_[np.ones(len(a)),a]@B,p[0])+p[1]*(np.c_[np.ones(len(a)),a]@B)+p[2]))
 out.sort(key=lambda z:z[0]);print(key,[(o[0],o[1],o[2],o[3]) for o in out[:10]]);print(out[0][4]);return out
W={'q':q}; models={}
for pn,key in [('b','v'),('c','v1'),('c','v2')]:
 opts=fit(s['panel_'+pn],key);name='b' if pn=='b' else key;models[name]=opts[0][-1]; W[name]=models[name](q)
for n,M in models.items():
 for p in models:
  if p!=n:W[p+n]=M(W[p])
np.savez('/tmp/panel.npz',**W)

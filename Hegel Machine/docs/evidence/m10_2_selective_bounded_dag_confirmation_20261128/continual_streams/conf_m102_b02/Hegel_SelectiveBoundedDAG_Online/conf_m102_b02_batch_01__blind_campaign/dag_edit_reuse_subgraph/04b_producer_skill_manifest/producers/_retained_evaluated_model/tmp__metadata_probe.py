import contextlib,io,itertools
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/compare_maps.py').read())
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.optimize import brentq
from node00_baseline import _latent

# Panel-only alternatives: jointly fit a common distortion law and its mixing matrix.
laws={'tanh':lambda x,k:np.tanh(k*x),'rat':lambda x,k:x/(1+k*x*x),'atan':lambda x,k:np.arctan(k*x),'asinh':lambda x,k:np.arcsinh(k*x),'cos':lambda x,k:np.cos(k*x)-1,'sin':lambda x,k:np.sin(k*x)}
def panels(s):
 out={}
 for nm,kk,ky in [('B','panel_b','v'),('C1','panel_c','v1'),('C2','panel_c','v2')]:
  p=s[kk]; U,V=p['u'],p[ky]; f=laws[{'B':'rat','C1':'tanh','C2':'cos'}[nm]]
  def fit(k):
   F=np.c_[np.ones(len(U)),U,f(U,k)]; C=np.linalg.lstsq(F,V,rcond=None)[0];return np.mean((F@C-V)**2),C
  ks=np.linspace(.2,4,48);i=np.argmin([fit(k)[0] for k in ks]);k=minimize_scalar(lambda k:fit(k)[0],bounds=(ks[max(0,i-1)],ks[min(i+1,len(ks)-1)]),method='bounded').x
  _,C=fit(k); A=C[1:5]; L,_,R=np.linalg.svd(A); M=L@R; b=C[0]; W=C[5:]; r=np.sum(A*W)/np.sum(A*A)
  def inv(t,A=A,b=b,k=k,r=r,f=f):
   q=np.linalg.solve(A.T,(t-b).T).T; grid=np.linspace(-4,4,4097); val=grid+r*f(grid,k)
   ix=np.argsort(val,kind='stable'); return np.interp(q,val[ix],grid[ix])
  out.update({nm:lambda t,A=A,b=b,k=k,W=W,f=f:t@A+f(t,k)@W+b,nm+'i':inv,nm+'L':lambda t,A=A,b=b:t@A+b,nm+'R':lambda t,M=M:t@M,nm+'T':lambda t,M=M:t@M.T,nm+'pre':lambda t,r=r,k=k,f=f:t+r*f(t,k),nm+'post':lambda t,r=r,k=k,f=f,M=M:(t@M+r*f(t@M,k))@M.T})
 out['I']=lambda t:t
 return out

def source(s,mode):
 z,n=_latent(s); p=s['train']; q=s['test']; j=mode[0]
 if j=='latent':return z,n
 if j=='x':return np.vstack([p['x'],q['x']]),n
 if j in p:
  v=np.asarray(p[j],float)
  if v.shape==(z.shape[1],):
   if mode[1]=='offset':z=z-v
   elif mode[1]=='add':z=z+v
   elif mode[1]=='scale':z=z*np.maximum(abs(v),.02)
   elif mode[1]=='divide':z=z/np.maximum(abs(v),.02)
 return z,n

if __name__=='__main__':
 Z,n=_latent(s); X=Z[:n]; funcs=panels(s); results=[]
 srcs=[('latent','I'),('x','I')]+[(k,op) for k,v in s['train'].items() if np.shape(v)==(X.shape[1],) for op in ('offset','add','scale','divide')]
 for src in srcs:
  x0=source(s,src)[0][:n]
  for j,f in funcs.items():
   h1=f(x0)
   keys=funcs if src==('latent','I') else {'I':funcs['I']}
   for k,g in keys.items():
    q=g(h1); sd=max(.0001,np.mean(q.std(0)))
    for sc in [.08,.13,.20,.32,.52,.85]:
     e=score(q*(sc/sd)); results.append((max(e),e.tolist(),src,j,k,sc))
 print('Panel-related coordinate candidates (proxy):',*sorted(results)[:7],sep='\n',flush=True)
 print('Public proxy:',*sorted(results,key=lambda r:max(r[1][:2]))[:5],sep='\n',flush=True)
 import ast
 opts=sorted(results)[:3]+[min(results,key=lambda r:max(r[1][:2]))]
 text=open('/tmp/metadata_probe.py').read(); text=text[text.index('from scipy.optimize import minimize_scalar'):text.index("if __name__=='__main__':")];text=text.replace('from node00_baseline import _latent',open('/tmp/node00_baseline.py').read().split('def run_node')[0]);
 for idx,(_,_,src,j,k,sc) in enumerate(opts):
  body='\ndef run_node(inputs: dict) -> dict:\n    study=inputs["__input__"]\n    q,n=source(study,'+repr(src)+')\n    f=panels(study)\n    q=f['+repr(k)+'](f['+repr(j)+'](q))\n    q=q*('+repr(sc)+'/max(.0001,np.mean(q[:n].std(0))))\n    return {"z":q[:n].copy(),"zt":q[n:].copy()}\n';open('/tmp/node00_panel_'+str(idx)+'.py','w').write(text+body)

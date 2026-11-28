exec(open('/tmp/dag.py').read().split('ops=')[0])
from scipy.optimize import least_squares
from itertools import product
n=52; a,b=Q(z),R(z); cross=(a[:,:,None]*b[:,None,:]).reshape(-1,16); F=ones(np.c_[a,b,cross]); Y=np.c_[y,ts]; U,sig,V=np.linalg.svd(ts[:n],full_matrices=False); M=ts@V[:3].T; target=np.c_[y,M]; print('cond coef',ls(np.c_[F[:n],M[:n]],y[:n])); print('iv ranks',np.linalg.svd((target-F@ls(F[:n],target[:n]))[:n],compute_uv=False));
rng=np.random.default_rng(1234)
blocks={'q':a,'r':b,'z':z,'l':L(z),'qr':np.c_[a,b],'q+r':a+b,'q-r':a-b,'q*r':a*b,'dot':(a*b).sum(1)[:,None]};
# Keep base controls while looking for nonlinear gains on a linear or bilinear numerator.
bases={'qr':ones(np.c_[a,b]),'qc':ones(a),'rc':ones(b),'full':F,'one':np.ones((len(z),1)),'z':ones(z),'diag':ones(np.c_[a,b,a*b])}
funcs={'tanh':np.tanh,'sin':np.sin,'gauss':lambda v:np.exp(-v*v),'rat':lambda v:1/(1+v*v),'soft':lambda v:v/(1+abs(v))};
for out in [0,1,2,3]:
 t=target[:,out]; best=[]
 for cn,base in bases.items():
  for dn in ['one','qc','rc','qr','diag','z'] if cn in ['qr','one','diag'] else ['one']:
   num=bases[dn]; P=np.linalg.qr(base[:n])[0]; ry=t[:n]-P@(P.T@t[:n]);
   for wn,w0 in blocks.items():
    w=(w0-w0[:n].mean(0))/np.maximum(w0[:n].std(0),1e-8); d=w.shape[1]; u=ls(w[:n],ry); u=u/max(np.linalg.norm(u),1e-8)
    for gn,g in funcs.items():
     def mat(p,rows):return num[rows]*g(w[rows]@p[:-1]+p[-1])[:,None]
     def res(p):
      m=mat(p,slice(n)); m=m-P@(P.T@m); return ry-m@ls(m,ry)
     starts=[np.r_[u,0.2], np.r_[u*2,.7],np.r_[rng.normal(0,.6,d),-.5]]
     trials=[least_squares(res,p,max_nfev=55,bounds=(-12,12)) for p in starts]; p=min(trials,key=lambda v:v.cost).x; m=np.c_[base,mat(p,slice(None))]; coef=ls(m[:n],t[:n]); e=abs(m@coef-t);
     best.append((e[:n].mean(),e[n:].mean(),cn,dn,wn,gn,p,coef))
 print('output',out,flush=True)
 for v in sorted(best,key=lambda v:v[0])[:5]+sorted(best,key=lambda v:v[1])[:6]:print(v,flush=True)

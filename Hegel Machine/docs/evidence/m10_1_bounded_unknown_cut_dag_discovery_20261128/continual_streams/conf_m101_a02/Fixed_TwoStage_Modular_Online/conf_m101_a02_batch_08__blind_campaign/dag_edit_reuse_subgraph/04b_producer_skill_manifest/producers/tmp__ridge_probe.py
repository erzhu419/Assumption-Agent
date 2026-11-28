exec(open('/tmp/readout_probe.py').read().split('np.set_printoptions')[0])
from scipy.optimize import least_squares
from scipy.linalg import qr
np.set_printoptions(precision=4,suppress=True)
D=poly(z,2); C=np.linalg.lstsq(D,Y,rcond=None)[0]; E=Y-D@C
print('quadratic coefficients label/final',C[:,[0,-1]].T)
print('singulars after affine, quadratic',np.linalg.svd(Y-poly(z,1)@np.linalg.lstsq(poly(z,1),Y,rcond=None)[0],compute_uv=False),np.linalg.svd(E,compute_uv=False))
for j in [0,-1]:
 H=np.zeros((4,4));
 for c,(a,b) in zip(C[5:,j],itertools.combinations_with_replacement(range(4),2)):
  H[a,b]=H[b,a]=c if a==b else c/2
 print('quadratic eigensystem',np.linalg.eigh(H))
print('contrast',check(poly(z,1),(Y[:,0]-Y[:,-1])[:,None]) if False else np.std(np.linalg.lstsq(poly(z,1),Y[:,0]-Y[:,-1],rcond=None)[0]@poly(z,1).T-(Y[:,0]-Y[:,-1])))
F={'sin':np.sin,'tanh':np.tanh,'gauss':lambda t:np.exp(-t*t),'logcosh':lambda t:np.logaddexp(t,-t),'rational':lambda t:1/(1+t*t),'atan':np.arctan,'cubic':lambda t:t**3,'exp':lambda t:np.exp(np.clip(t,-20,20))}
def fit(z,Y,names,starts=8,degree=1):
 D=poly(z,degree); Q=np.linalg.qr(D)[0]; B=Y-Q@(Q.T@Y)
 rng=np.random.default_rng(42)
 def feats(z,p):return np.column_stack([F[n](q[4]+z@q[:4]) for q,n in zip(p.reshape(-1,5),names)])
 def res(p):
  N=feats(z,p); N-=Q@(Q.T@N);return (B-N@np.linalg.lstsq(N,B,rcond=1e-9)[0]).ravel()
 best=None
 for _ in range(starts):
  p=least_squares(res,rng.normal(0,0.9,5*len(names)),bounds=(-8,8),max_nfev=130)
  if best is None or p.cost<best.cost:best=p
 p=best.x; return best.cost,p,lambda v:np.c_[poly(v,degree),feats(v,p)]@np.linalg.lstsq(np.c_[D,feats(z,p)],Y,rcond=1e-9)[0]
if __name__=='__main__':
 a,b=fs[0]; models=[]
 for d in [1,2]:
  for n in list((f,) for f in F)+list(itertools.combinations_with_replacement(['sin','tanh','gauss','logcosh','rational','atan'],2)):
   cost,p,pred=fit(z[a],Y[a],n,8,d);e=np.mean(abs(pred(z[b])-Y[b]),axis=0);models.append((cost,d,n,p,e[0],e[1:].mean()))
 for cost,d,n,p,ey,et in sorted(models,key=lambda x:x[0])[:20]:print('fit',round(cost,5),d,n,p,'cv',round(ey,5),round(et,5))

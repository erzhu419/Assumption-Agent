exec(open('/tmp/walls04.py').read().split('for a in range(3):')[0])
from itertools import product
from scipy.optimize import lsq_linear

def stencil(a, kind):
 R,P,O,h=axes[a]; k=s['kappa_'+'xy'[a]]; v=s['vel_'+'xy'[a]]; M=R.toarray()
 for i,j in [(0,1),(-1,-2)]:
  speed=max(-v[i],0) if i==0 else max(v[i],0)
  val={'archive':k[i]/h**2,'mirror':2*k[i]/h**2,'face':(k[i]+k[j])/(2*h**2),'mirror_face':(k[i]+k[j])/h**2,'upwind':(k[i]+k[j])/(2*h**2)+speed/h,'endpoint_upwind':k[i]/h**2+speed/h}[kind]
  M[i,i],M[i,j]=val,-val
 return sparse.csc_matrix(M)

def constrained(S, idx):
 A=S[idx]; B=Z[idx]; b=y[idx]; Q=np.linalg.lstsq(B,np.c_[A,b],rcond=None)[0]; D=A-B@Q[:,:-1]; t=b-B@Q[:,-1]
 w,_=robin._box_simplex(D.T@D+1e-12*np.eye(D.shape[1]),-D.T@t,s['u_box'],s['total_budget'])
 return w,Q[:,-1]-Q[:,:-1]@w

def score(S,tr,te):
 w,g=constrained(S,tr); return np.sqrt(np.mean((S[te]@w+Z[te]@g-y[te])**2))

kinds=('archive','mirror','face','mirror_face','upwind','endpoint_upwind')
res=[]
for a,b in product(kinds,repeat=2):
 S=forward(stencil(0,a),stencil(1,b))
 cv=[score(S,full[sensors%4!=f],full[sensors%4==f]) for f in range(4)]
 cs=[score(S,full[shifts!=f],full[shifts==f]) for f in range(3)]
 res.append((max(cv),max(cs),score(S,full,full),a,b))
print('best alternative reflecting stencils (sensor worst, shift worst, pooled, x, y)')
for row in sorted(res)[:8]: print(row)
S=forward(stencil(0,'archive'),stencil(1,'archive'))
w,g=constrained(S,full); resid=S@w+Z@g-y
print('archive/archive',next(q for q in res if q[-2:]==('archive','archive')),'fit weights',w.round(4),'nuisance',g.round(4))
print('residual bank rms',np.sqrt(np.mean(resid.reshape(-1,4)**2,axis=0)).round(6),'rosette telemetry corr',np.corrcoef(np.c_[resid.reshape(-1,4),s['aux_manifold_telemetry']].T)[:4,4:].round(2))

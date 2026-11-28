exec(open('/tmp/scalar_probe.py').read().split('models=')[0])
from scipy.optimize import minimize, least_squares
from scipy.linalg import lstsq
from scipy.stats import pearsonr
for kind,nm in [('tanh','v1'),('rational','v2')]:
 op=n._operator(c['u'],c[nm],kind,(.3,3.) if kind=='tanh' else (.015,.8)); print(nm,'fit RMS',np.sqrt(np.mean((op(c['u'])-c[nm])**2)))
for label,v in [('x',x),('z',z),('sy',sy),('st',st[:,:4])]:
 f=np.c_[np.ones(N),poly(v,2)]; coef=np.linalg.lstsq(f,y,rcond=None)[0]; e=y-f@coef; print(label,'quad',np.round(coef,3),'res',np.sqrt(np.mean(e**2)))
 print('res correlations cubes',np.round(np.corrcoef(e,poly(v,3)[:,14:].T)[0,1:],2))
# Kernel families available from handoff; select hyperparameters strictly within each outer training fold.
D=lambda v:cdist(v,v,'sqeuclidean')
syd=D(sy); std=D(st[:,:4]); kd=np.maximum(0,K.diagonal()[:,None]+K.diagonal()[None,:]-2*K)
base=[('K',K),('stq',K+30*st@st.T),('sy2',K+.3*poly(sy,2)@poly(sy,2).T)]
for label,d in [('sy',syd),('st',std),('Kdist',kd)]:
 for l in [2,6,12,24,48,96]:
  for a in [.03,.3,3]: base.append((f'{label}_{l}_{a}',K+a*np.exp(-d/l)))
for label,v in [('x',x),('z',z),('sy',sy),('st',st[:,:4])]:
 G=sum((v@v.T)**k / np.math.factorial(k) if hasattr(np,'math') else (v@v.T)**k / __import__('math').factorial(k) for k in range(1,4));
 for a in [.03,.3,3,30]:base.append((f'{label}_smoothpoly_{a}',K+a*G))
configs=[(k,g,r) for k,g in base for r in [1e-5,3e-5,1e-4,3e-4,.001,.003]]
for item in sorted([(cv(g,r),k,r) for k,g,r in configs],key=lambda m:.5*m[0][1]+.25*m[0][2]+.25*m[0][3])[:18]:print(item)
for te in folds:
 tr=np.setdiff1d(np.arange(N),te); yy=y[tr]; mu=yy.mean(); scores=[]
 for name,g,r in configs:
  H=solve(g[np.ix_(tr,tr)]+r*np.eye(len(tr)),np.eye(len(tr)),assume_a='pos'); al=H@(yy-mu); e=al/(H.diagonal()-(H@np.ones(len(tr)))/len(tr)); scores.append((np.mean(abs(e)),name,r,g,al))
 top=sorted(scores,key=lambda z:z[0])[:5]; print('nested',[(round(a,5),k,r,round(np.mean(abs(g[np.ix_(te,tr)]@al+mu-y[te])),5)) for a,k,r,g,al in top[:3]],'blend',np.mean(abs(np.mean([g[np.ix_(te,tr)]@al+mu for _,k,r,g,al in top],axis=0)-y[te])))

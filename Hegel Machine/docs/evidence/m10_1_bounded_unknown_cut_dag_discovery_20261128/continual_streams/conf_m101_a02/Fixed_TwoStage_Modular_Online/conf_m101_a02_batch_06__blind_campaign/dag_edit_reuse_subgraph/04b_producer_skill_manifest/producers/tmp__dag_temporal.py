import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/dag_sink_probe.py').read())
from itertools import combinations_with_replacement, combinations
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.stats import spearmanr
np.set_printoptions(precision=4,suppress=True,linewidth=160)
base=stack(one(z),z); D=stack(base,Q); print('n',len(z)); print('all train summary',s['train']['summary'])
for stage,H in [('full',tr),('linear removed',tr-base@ls(base,tr)),('quad removed',tr-D@ls(D,tr))]:
 _,sv,vt=np.linalg.svd(H,full_matrices=False); print(stage,sv)
 funcs={'log':lambda k: np.log1p(k*t),'pow':lambda k:t**k,'exp':lambda k:-np.expm1(-k*t),'expdec':lambda k:np.exp(-k*t),'rat':lambda k:t/(1+k*t),'sin':lambda k:np.sin(k*t),'sqrt':lambda k:np.sqrt(1+k*t)-1,'ones':lambda k:np.ones_like(t),'t':lambda k:t}
 def ev(p,i,j):
  F=stack(funcs[i](p[0]),funcs[j](p[1])); return (H-(H@np.linalg.pinv(F.T))@F.T).ravel()
 out=[]
 for i,j in combinations(funcs,2):
  opt=least_squares(ev,[1.,.7],args=(i,j),bounds=(.005,5)); out.append((np.std(opt.fun),i,j,opt.x))
 print('time fits',sorted(out,key=lambda x:x[0])[:10]);
for label,W in [('z',z),('A',A),('B',B),('ref',runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'))]:
 if label=='ref':
  m=W; p=s['panel_a']; cc,se=m['_fit_a'](p['u'],p['v']); W=m['_invert_a'](s['train']['x'],cc,se)
 def quadratic(F): return stack(*[F[:,i]*F[:,j] for i,j in combinations_with_replacement(range(F.shape[1]),2)])
 def evaluate(F,a=0):
  X=stack(D,F); X=X/np.r_[1,np.maximum(X[:,1:].std(0),1e-8)]; gt=X@X.T; out=[]
  for i in range(6):
   ii=np.arange(len(z)); h=ii>=52 if i==0 else ii%5==i-1; K=gt[~h][:,~h]+(a+1e-10)*np.eye(sum(~h)); E=abs(gt[h][:,~h]@solve(K,Y[~h],assume_a='pos')-Y[h]); out.append([E[:,0].mean(),E[:,1:].mean()])
  return np.r_[out[0],np.mean(out[1:],axis=0),np.max(out[1:],axis=0)]
 fs={'rawquad':quadratic(W),'sqrtabs':np.sign(W)*np.sqrt(abs(W)),'relu2':quadratic(abs(W)),'zabs':quadratic(stack(W,abs(W))),'z+c':quadratic(stack(W,np.tanh(1.7*W))),'cos-quad':quadratic(np.cos(2*W)),'tanhquad':quadratic(np.tanh(1.7*W)),'sinquad':quadratic(np.sin(2*W)),'quad-rat':quadratic(W/(1+2*W**2))}
 out=[]
 for nm,F in fs.items():
  for a in [.003,.1,1,5]: out.append((evaluate(F,a),nm,a))
 print(label,sorted(out,key=lambda p:max(p[0][2:4]))[:6])

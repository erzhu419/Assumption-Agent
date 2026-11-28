exec(open('/tmp/probe.py').read().split('ks=')[0])
from scipy.optimize import minimize_scalar
print('archive diagnostics',open('/tmp/hegel_continual_archive/archive_manifest.json').read()[:2600]); print(open('/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json').read()[:5000]); print('ranges U',U.min(0),U.max(0)); print('cal error quantiles',np.quantile(np.linalg.norm(n._forward(s['panel_a']['u'],n._calibrate(s['panel_a'],f))-s['panel_a']['v'],axis=1),[0,.5,.8,.9,1])); print('T correlation',np.round(np.corrcoef(T.T)[0],4))
fun={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'inv3':lambda x:np.sinh(np.arcsinh(x)/3),'rat':lambda x:x/(1+x*x),'sat':lambda x:x/np.sqrt(1+x*x),'r2':lambda x:1/(1+x*x),'soft':lambda x:np.sqrt(1+x*x)}
def aff(x):return np.c_[np.ones(len(x)),x]
def pair(x):return np.stack([x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(x.shape[1]),2)],axis=1)
def solve(x,y,idx):return np.linalg.lstsq(x[idx],y[idx],rcond=1e-10)[0]
def fit(x,y,k,extra=False):
 fn=fun[k]; idx=np.arange(len(x))<len(x)*.76
 def d(g):return np.c_[aff(x),fn(g*x)] if extra else aff(fn(g*x))
 def loss(g):
  z=d(g); return np.mean((z[idx]@solve(z,y,idx)-y[idx])**2)
 g=minimize_scalar(loss,bounds=(.05,7),method='bounded').x; z=d(g); q=solve(z,y,idx); e=z@q-y
 return np.round([g,np.abs(e[idx]).mean(),np.abs(e[~idx]).mean()],5),lambda x: (np.c_[aff(x),fn(g*x)] if extra else aff(fn(g*x)))@q
for name,p,y in [('B',b,b['v']),('C1',c,c['v1']),('C2',c,c['v2'])]:
 x=p['u']; print(name, 'aff residual',np.round(np.std(y-aff(x)@np.linalg.lstsq(aff(x),y,rcond=None)[0],axis=0),6))
 print('transforms',[(k,fit(x,y,k,extra)[0].tolist()) for extra in [False,True] for k in fun]); print('quadratic coefficients',np.round(np.linalg.lstsq(np.c_[aff(x),pair(x)],y,rcond=None)[0],3))
B=fit(b['u'],b['v'],'sin',True)[1]; C1=fit(c['u'],c['v1'],'asinh',True)[1]; C2=lambda x:np.c_[aff(x),pair(x)]@np.linalg.lstsq(np.c_[aff(c['u']),pair(c['u'])],c['v2'],rcond=None)[0];
V={'U':U,'W':W,'Bfit':B(U),'C1':C1(U),'C2':C2(U)}
for key,x in list(V.items()):
 for kk in ('cos','r2','soft'):
  print(key,kk,fit(x[:72],np.c_[Y,T],kk,True)[0]);
for ys in [Y,T]:
 for k,x in V.items():
  for fe in [aff(x),np.c_[aff(x),pair(x)],np.c_[aff(x),*[fn(2.15*x) for fn in (np.sin,np.cos,np.tanh)]],np.c_[aff(x),pair(np.tanh(2.15*x)),pair(np.sin(2.15*x))]]:
   idx=np.arange(72)<52; er=fe[:72]@solve(fe[:72],ys,idx)-ys; print(k,fe.shape[1],round(abs(er[idx]).mean(),4),round(abs(er[~idx]).mean(),4))

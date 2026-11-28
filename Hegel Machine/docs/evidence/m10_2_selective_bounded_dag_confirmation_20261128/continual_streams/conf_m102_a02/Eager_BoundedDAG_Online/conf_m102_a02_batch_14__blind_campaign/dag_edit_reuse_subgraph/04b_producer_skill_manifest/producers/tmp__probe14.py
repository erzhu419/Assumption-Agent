import numpy as np,pickle,sys,itertools
from scipy.special import erf
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/tmp/hegel_continual_archive')
import conf_m102_a02_batch_06_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=np.c_[t['y'],t['traj']]
def poly(x,k):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for d in range(1,k+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),d)])
def ols(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
p=s['panel_a']; u,v=p['u'],p['v']; mask=np.max(abs(v),axis=1)<4
for i in range(4):
 print('a',i,[(j,round(np.std((lambda a: a@ols(a,v[mask,i])-v[mask,i])(np.c_[np.ones(mask.sum()),u[mask,i],u[mask,j],u[mask,i]*u[mask,j],u[mask,i]**3])),5)) for j in range(4) if i!=j])
z=old._calibrate(p,x); print('inverse range',z.min(),z.max(),'robust round trip inverse',np.max(abs(old._calibrate(p,v[mask])-u[mask])))
def fit_panel(p,z,n):
 u,v=p['u'],p[n]; funcs={'tanh':np.tanh,'erf':erf,'sin':np.sin,'rat':lambda x:x/(1+x*x),'sat': lambda x:x/np.sqrt(1+x*x),'cos':np.cos,'exp2':lambda x:np.exp(-x*x),'sq':np.square,'cube':lambda x:x**3,'softplus':lambda x:np.logaddexp(0,x)}
 results=[]
 for name,f in funcs.items():
  def design(a,w):return np.c_[poly(a,1),f(w*a)]
  def loss(w):
   a=design(u,w);return np.mean((a@ols(a,v)-v)**2)
  opt=minimize_scalar(loss,bounds=(.2,4),method='bounded'); a=design(u,opt.x); co=ols(a,v); h=np.einsum('ij,ji->i',a,np.linalg.pinv(a));loo=np.sqrt(np.mean(((a@co-v)/np.maximum(1-h[:,None],1e-5))**2)); results.append((loo,name,opt.x,opt.fun**.5,design(z,opt.x)@co))
 for k in (1,2):
  a=poly(u,k);co=ols(a,v);h=np.einsum('ij,ji->i',a,np.linalg.pinv(a));results.append((np.sqrt(np.mean(((a@co-v)/(1-h[:,None]))**2)),f'poly{k}',0,np.std(a@co-v),poly(z,k)@co))
 results.sort(key=lambda x:x[0]); print(n,[(r[1],*np.round([r[0],r[2],r[3]],5)) for r in results]); return results[0][-1]
g=fit_panel(s['panel_b'],z,'v'); j=fit_panel(s['panel_c'],z,'v1'); k=fit_panel(s['panel_c'],z,'v2');
def score(a):
 c=ols(a[:52],y[:52]);e=abs(a[52:]@c-y[52:]);return np.round([e[:,0].mean(),e[:,1:].mean()],5)
for tag,a in [('z',z),('g',g),('j',j),('k',k),('zg',np.c_[z,g]),('zj',np.c_[z,j]),('zk',np.c_[z,k]),('gj',np.c_[g,j]),('gk',np.c_[g,k]),('jk',np.c_[j,k]),('all',np.c_[z,g,j,k])]:
 print(tag,score(poly(a,1)),score(poly(a,2)))
print('old',[(abs(a-b)).mean() for a,b in zip(old._predict(z,old._panel(s['panel_b'],z),np.arange(52),np.arange(52,72),y[:52,0],y[:52,1:])[:2],(y[52:,0],y[52:,1:]))]);

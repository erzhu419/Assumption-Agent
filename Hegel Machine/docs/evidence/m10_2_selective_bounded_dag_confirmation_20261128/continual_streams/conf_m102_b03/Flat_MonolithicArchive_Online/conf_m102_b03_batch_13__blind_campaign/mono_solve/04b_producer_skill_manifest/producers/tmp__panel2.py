import pickle,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u=s['panel_c']['u']; y=s['panel_c']['v2'];D=lambda a:np.c_[np.ones(len(a)),a]; L=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]; q=[u*np.roll(u,j,axis=1) for j in range(4)];out=[]
fs={'cos':np.cos,'sin':np.sin,'tsin':lambda x:x*np.sin(x),'sat':lambda x:x/(1+abs(x)),'rational':lambda x:x/(1+x*x),'evenrat':lambda x:1/(1+x*x),'logcosh':lambda x:np.logaddexp(x,-x),'exp':np.exp,'tanh':np.tanh,'arctan':np.arctan,'abs':np.abs}
def score(a,loo=False):
 a=D(a); p=np.linalg.pinv(a,rcond=1e-10);r=y-a@(p@y);return np.sqrt(np.mean((r/(np.maximum(.01,1-np.einsum('ij,ji->i',a,p))[:,None] if loo else 1))**2))
for k,f in fs.items():
 for a,base in [('none',u[:,:0]),('lin',u)]+[(f'q{j}',v) for j,v in enumerate(q)]+[(f'linq{j}',np.c_[u,v]) for j,v in enumerate(q)]:
  for j,arg in [('u',u)]+[(f'pair{j}',v) for j,v in enumerate(q)]+[(f'sum{j}',u+np.roll(u,j,axis=1)) for j in range(1,4)]+[(f'diff{j}',u-np.roll(u,j,axis=1)) for j in range(1,4)]:
   def des(t):return np.c_[base,f(t*arg)]
   opt=minimize_scalar(lambda t: score(des(t),True), bounds=(.12,4.5),method='bounded');out.append((opt.fun,score(des(opt.x)),k,a,j,opt.x))
print(*sorted(out)[:22],sep='\n')
print('regular pair features:',[(j,score(np.c_[u,q[j]],True),score(q[j],True)) for j in range(4)])

import numpy as np,pickle
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_a'][i] for i in ('u','v')]; k=s['telemetry_vec'][1]; d=u.shape[1]; dsg=lambda x:np.c_[np.ones(len(x)),x]
def f(D,v):return np.linalg.lstsq(D,v,rcond=None)[0]
a=dsg(u); e=np.linalg.norm(a@f(a,v)-v,axis=1); g=e<5
for _ in range(6):
 c=f(a[g],v[g]); e=np.linalg.norm(a@c-v,axis=1); g=e<max(.002,4*np.median(e))
print('affine coef',c,'errors',np.round(e,4),'exclude',np.where(~g)[0]);print('ranges',[(x.min(0),x.max(0)) for x in (u,v,s['train']['x'],s['test']['x'])]); print('panel A values after excluding gross outliers:');print(np.round(np.c_[u,v,e][g][:14],4))
funcs={'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'sin':np.sin,'cos':np.cos,'erf':erf,'square':lambda x:x*x,'cube':lambda x:x**3,'abs':np.abs,'softsign':lambda x:x/(1+abs(x)),'rational':lambda x:x/(1+x*x),'sqrt':lambda x:x/np.sqrt(1+x*x),'softplus':lambda x:np.logaddexp(x,0),'sinh':np.sinh}
g=e<5
import itertools
pairs=np.array(list(itertools.combinations(range(d),2))); i,j=pairs.T;p=u[:,i]*u[:,j]; plus=u[:,i]+u[:,j]; minus=u[:,i]-u[:,j]; cyc=u*np.roll(u,-1,axis=1)
Ds={'affine':a,'archive poly':np.c_[a,cyc,u**3],'quadratic':np.c_[a,u**2,p],'odd poly':np.c_[a,u**3]}
for name,func in funcs.items():
 for lab,t in [('u',u),('sum/diff',np.c_[plus,minus]),('prod',p),('post affine',a@c)]: Ds[name+' '+lab]=np.c_[a,func(k*t)]
res=[]
for name,D in Ds.items():
 co=f(D[g],v[g]); e0=np.linalg.norm(D@co-v,axis=1); hat=np.sum(np.linalg.qr(D[g])[0]**2,axis=1); loo=e0[g]/(1-hat);res.append((np.mean(loo),np.max(loo),np.mean(e0[g]),name))
print('best panel A calibration loo:');print(*sorted(res)[:35],sep='\n')
b=s['panel_b']; aa=dsg(b['u']); print('panel B residual',np.max(abs(aa@f(aa,b['v'])-b['v'])))

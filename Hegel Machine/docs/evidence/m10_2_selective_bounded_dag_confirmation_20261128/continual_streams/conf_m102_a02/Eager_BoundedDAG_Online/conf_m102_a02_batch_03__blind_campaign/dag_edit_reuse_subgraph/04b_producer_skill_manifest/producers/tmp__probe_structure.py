exec(open('/tmp/probe_dag.py').read().split('for k in (')[0])
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist

def temporal(p):
 return -np.expm1(-np.exp(p)[:,None]*t)
def resid(p, z):
 h=temporal(p); return (np.linalg.lstsq(h.T,z.T,rcond=None)[0].T@h-z).ravel()
p=least_squares(resid,np.log([.4,1.8]),args=(tr[:52],)).x; h=temporal(p); w=np.linalg.lstsq(h.T,tr.T,rcond=None)[0].T
print('rates',np.exp(p),'MAE',np.mean(abs(resid(p,tr))),'amplitudes std',w.std(0));print('affine amplitude',np.linalg.lstsq(D(x,1),w,rcond=None)[0]);
pa=s['panel_a']; print('panel A u | v-u:'); print(np.round(np.c_[pa['u'],pa['v']-pa['u']],3));
pb=s['panel_b']; cb=np.linalg.lstsq(D(pb['u'],1),pb['v'],rcond=None)[0]; u=(x-cb[0])@np.linalg.inv(cb[1:]); print('inverse panel b ranges',u.min(0),u.max(0)); print('direct panel b ranges',(D(x,1)@cb).min(0),(D(x,1)@cb).max(0));
for name,xx in [('raw',x),('invb',u),('b',D(x,1)@cb)]:
 print(name,'quad coefficients y | amplitudes');print(np.linalg.lstsq(D(xx,2),np.c_[y,w],rcond=None)[0].round(3))
print('public/kernel tests; compare mean over five test blocks and last block')
blocks=np.array_split(np.arange(len(x)),5); blocks.append(np.arange(52,72)); yy=np.c_[y,tr]
def score(K,reg):
 a=[]
 for j in blocks:
  i=np.setdiff1d(np.arange(len(x)),j); p=K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+reg*np.eye(len(i)),yy[i]);e=abs(p-yy[j]); a.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.array(a)
for gam in [.03,.08,.16,.32,.64,1.28]:
 dist=cdist(x,x,'sqeuclidean'); dot=x@x.T; r=np.sqrt(dist*gam); q=D(x,2); q=q@q.T
 for name,K in [('rbf',np.exp(-gam*dist)),('matern', (1+r+r*r/3)*np.exp(-r)),('poly',(1+gam*dot)**3),('rbf+quad',np.exp(-gam*dist)+q)]:
  for reg in [.000003,.0001,.001]:
   a=score(K,reg); print(gam,name,reg,'cv mean',a[:5].mean(0).round(4),'worst',a[:5].max(0).round(4),'official',a[-1].round(4))

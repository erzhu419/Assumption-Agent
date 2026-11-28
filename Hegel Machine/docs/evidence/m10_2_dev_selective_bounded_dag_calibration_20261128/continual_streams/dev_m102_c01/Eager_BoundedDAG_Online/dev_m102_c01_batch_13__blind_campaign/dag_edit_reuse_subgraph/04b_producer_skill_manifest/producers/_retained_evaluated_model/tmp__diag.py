exec(open('/tmp/readout.py').read().split('for name,f in dict')[0])
from scipy.spatial.distance import cdist
from scipy.linalg import solve
sl,off=a._calibration(s['panel_a']); B=s['panel_b']; C=s['panel_c']; bw=np.linalg.lstsq(np.c_[np.ones(len(B['u'])),B['u']],B['v'],rcond=None)[0]; r=((x-off)/sl)@bw[1:]+bw[0]; r2=(x*sl+off)@bw[1:]+bw[0]
for nm,d in [('x',x),('r',r),('r2',r2),('sy',u),('st',v)]:
 F=np.c_[np.ones(n),poly(d,2)]; w=np.linalg.lstsq(F,y,rcond=None)[0]; e=y-F@w; print(nm,'coef',w.round(3),'res',np.quantile(abs(e),[0,.25,.5,.75,.9,.95,1]).round(4))
print('rows high',[(i,x[i].round(2),round(y[i],3)) for i in np.argsort(abs(e))[-8:]])
folds=[np.arange(n-20,n)]+list(np.array_split(np.arange(n),4))
def cv(H,lam):
 out=[]
 for j in folds:
  i=np.setdiff1d(np.arange(n),j); e=y[j]-H[np.ix_(j,i)]@solve(H[np.ix_(i,i)]+lam*np.eye(len(i)),y[i],assume_a='pos'); out.append(np.mean(abs(e)))
 return np.array(out)
models=[]
for nm,d in [('x',x),('r',r),('r2',r2),('sy',u),('st',v),('mix',np.c_[u,v])]:
 d=d/(d.std(0)+1e-8); Q=1+d@d.T; D=cdist(d,d,'sqeuclidean');
 for base in [Q**2,p['K'],1.]:
  for k in [.04,.12,.35,1.]:
   for alpha in [.1,1,10]:
    for l in [.00001,.0003,.003,.03]:
     H=base+alpha*np.exp(-k*D); es=cv(H,l); models.append((max(es),es.mean(),es,nm,0 if np.isscalar(base) else (1 if base is p['K'] else 2),k,alpha,l))
for row in sorted(models,key=lambda r:r[0])[:20]:print(row)
for j in folds:
 i=np.setdiff1d(np.arange(n),j); H=p['K']; pred=H[np.ix_(j,i)]@solve(H[np.ix_(i,i)]+.0001*np.eye(len(i)),y[i],assume_a='pos'); print('kernel worst',[(int(j[k]), x[j[k]].round(2),round(y[j[k]]-pred[k],3)) for k in np.argsort(abs(y[j]-pred))[-5:]])

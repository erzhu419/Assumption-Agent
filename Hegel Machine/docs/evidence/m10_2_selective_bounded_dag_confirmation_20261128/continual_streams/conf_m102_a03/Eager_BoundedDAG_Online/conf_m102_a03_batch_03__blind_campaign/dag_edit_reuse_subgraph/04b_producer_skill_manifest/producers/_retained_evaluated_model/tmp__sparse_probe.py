exec(open('/tmp/explore.py').read().split("for k in ('panel_a'")[0])
from sklearn.linear_model import Lasso, MultiTaskLasso
B=poly(x,3); names=[''.join(map(str,ii)) or '1' for k in range(4) for ii in itertools.combinations_with_replacement(range(4),k)]; out=np.c_[y,t]; split=np.arange(52); val=np.arange(52,72); fs=np.array_split(np.random.default_rng(42).permutation(72),6)
for j in [0,1,4,8]:
 w=np.linalg.lstsq(B,out[:,j],rcond=None)[0]; print('largest',j,sorted(zip(names,w),key=lambda z:-abs(z[1]))[:20]); print('resid',np.std(B@w-out[:,j]))
for name,z in [('x',x),('b_aff',aff(x)@np.linalg.lstsq(aff(s['panel_b']['u']),s['panel_b']['v'],rcond=None)[0]),('c1_aff',aff(x)@np.linalg.lstsq(aff(s['panel_c']['u']),s['panel_c']['v1'],rcond=None)[0]),('c2_aff',aff(x)@np.linalg.lstsq(aff(s['panel_c']['u']),s['panel_c']['v2'],rcond=None)[0])]:
 for basis in ['odd','full','trig','paircubic']:
  b=poly(z,3); mask=np.array([len(k) in (1,3) or k=='1' for k in names]); b=b[:,mask] if basis=='odd' else b[:,[len(set(k))<3 for k in names]] if basis=='paircubic' else np.c_[aff(z),np.sin(2*z),np.cos(2*z),poly(z,2)[:,5:]] if basis=='trig' else b
  for alpha in [0,0.001,0.004,0.01]:
   ee=[]
   for q in fs+[val]:
    p=np.setdiff1d(np.arange(72),q); scale=np.maximum(b[p].std(0),1e-10); scale[0]=1.; bb=b/scale
    if alpha: yh=Lasso(alpha=alpha,max_iter=5000).fit(bb[p,1:],y[p]).predict(bb[q,1:]); th=MultiTaskLasso(alpha=alpha*2,max_iter=5000).fit(bb[p,1:],t[p]).predict(bb[q,1:]); ph=np.c_[yh,th]
    else: ph=bb[q]@np.linalg.lstsq(bb[p],out[p],rcond=None)[0]
    r=abs(ph-out[q]);ee.append([r[:,0].mean(),r[:,1:].mean()])
   if max(ee[-1])<.055 or name=='x' and basis=='odd':print(name,basis,alpha,np.max(ee,0),ee[-1])

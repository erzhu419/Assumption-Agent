exec(open('/tmp/readout.py').read().split('for name,f in dict')[0])
sl,off=a._calibration(s['panel_a']); B=s['panel_b']; C=s['panel_c']; bw=np.linalg.lstsq(np.c_[np.ones(len(B['u'])),B['u']],B['v'],rcond=None)[0]; c1=a._operator(C['u'],C['v1'],'tanh',(.3,3.)); c2=a._operator(C['u'],C['v2'],'rational',(.015,.8)); print('panel B map',bw); print('telemetry',s['telemetry_vec']);
from scipy.optimize import minimize_scalar
for typ,vv,bds in [('tanh','v1',(.3,3)),('rational','v2',(.015,.8))]:
 def loss(k):
  f=a._features(C['u'],typ,k); return np.mean((C[vv]-f@np.linalg.lstsq(f,C[vv],rcond=None)[0])**2)
 k=minimize_scalar(loss,bounds=bds,method='bounded').x; f=a._features(C['u'],typ,k); print(typ,k,loss(k),np.round(np.linalg.lstsq(f,C[vv],rcond=None)[0],4))
res=[]
for xx,zz in [('raw',x),('inv',(x-off)/sl),('fwd',x*sl+off)]:
 for bb,rr in [('',zz),('B',zz@bw[1:]+bw[0]),('bInv',(zz-bw[0])@np.linalg.inv(bw[1:]))]:
  for cc,hh in [('',rr),('c1',c1(rr)),('c2',c2(rr))]:
   for dd in [1,2,3]:
    f=poly(hh,dd)
    for name,g in [('p',f),('p+t1',np.c_[f,z[:,0]]),('p+sy',np.c_[f,u]),('p+sy+t1',np.c_[f,u,z[:,0]])]:
     tr,oo,_=ev('',g); res.append((max(oo),tr,oo,xx+bb+cc,dd,name))
for r in sorted(res,key=lambda x:x[0])[:30]:print(r)
print('kernel basis sweeps')
from scipy.linalg import solve
for terms in ['K','K+sy','K+st','K+score','K+sy+score','st+sy+score','K+sy2+score']:
 for ridge in [.0001,.003,.03,.2,1,3]:
  H=np.ones((n,n));
  for nm,F in [('sy',u),('sy2',poly(u,2)),('st',p['st']),('score',z)]:
   if nm in terms.split('+'):H+=F@F.T
  if terms.startswith('K'):H+=p['K'];
  ee=[]
  for k in [3,0,1,2]:
   j=np.arange(52,72) if k==3 else np.array_split(np.arange(n),3)[k]; i=np.setdiff1d(np.arange(n),j); ee.append(np.mean(abs(y[j]-H[np.ix_(j,i)]@solve(H[np.ix_(i,i)]+ridge*np.eye(len(i)),y[i],assume_a='pos'))))
  print(terms,ridge,np.round(ee,4))

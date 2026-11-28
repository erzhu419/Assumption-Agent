exec(open('/tmp/graph.py').read())
f=w; a=aff(np.c_[w,pair(w)]); n=52; y0=tr['y']; T=tr['traj'];
def report(d):
 ee=[]
 for v0 in [np.arange(52,72),np.arange(0,20),np.arange(26,46),np.random.default_rng(23).choice(72,20,replace=False)]:
  i=np.setdiff1d(np.arange(72),v0); e=d[v0]@ls(d[i],y0[i])-y0[v0]; ee.append(np.mean(abs(e)))
 return np.round(ee,5)
ww=ls(a[:52],Y[:52]); print('base co',ww[:,[0,1,4,8]].round(5)); print('base residual',np.percentile(abs(a[:72]@ls(a[:72],y0)-y0),[0,25,50,75,90,100]));
co1=ls(c1(u),c['v1']); c1s=lambda f:aff(pair(f)[:,[0,2,3,5]])@co1[[0,5,7,8,10]]
c2s=lambda f:aff(np.logaddexp(0,2.2*f))@ls(aff(np.logaddexp(0,2.2*u)),v)
for name,f in [('w',w),('C1',c1s(w)),('C2',c2s(w)),('C2C1',c2s(c1s(w))),('C1C2',c1s(c2s(w))),('z',q)]:
 Ds={'aff':f,'pairs':pair(f),'sq':f*f,'cube':f**3,'tanh':np.tanh(2.15*f),'sin':np.sin(2.15*f),'cos':np.cos(2.15*f),'rat':f/(1+1.25*f*f),'splus':np.logaddexp(0,2.15*f),'triples':np.array([np.prod(f[:,j],axis=1) for j in combinations(range(4),3)]).T}
 print(name)
 for typ,b0 in Ds.items():
  d=np.c_[a,b0]; print(typ,report(d),end='; ')
 print()
print('base',report(a)); print('T co',ls(a[:52],T[:52]).round(3)); print('targets range',y0.min(),y0.max())

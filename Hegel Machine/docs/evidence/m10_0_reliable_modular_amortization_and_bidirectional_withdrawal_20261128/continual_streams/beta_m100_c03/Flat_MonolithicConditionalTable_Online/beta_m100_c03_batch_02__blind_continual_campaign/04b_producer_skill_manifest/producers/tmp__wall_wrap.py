exec(open('/tmp/wall_probe.py').read().split('for name,p in')[0])
def pred(q):
 axes=[]
 for t,z in enumerate('xy'):
  k=s['kappa_'+z]; v=s['vel_'+z]; h=s['L'+z]/(len(k)-1); rob=s['robin_'+z]; a=m._axis(k,v,h,0.).tolil()
  for j,(b,n,o) in enumerate(((0,1,len(k)-1),(len(k)-1,len(k)-2,0))):
   scale,wrap,leak=q[t,j]; inward=scale*k[b]/h**2; seam=wrap*(k[b]+k[o])/(2*h*h)
   a[b,n]=-inward; a[b,o]=-seam; a[b,b]=inward+seam+leak*rob/h
  axes.append(a.tocsc())
 return m._responses(s,axes)[0]@w
for name,q in [('wrap both',[[1,1,0]]*4),('wrap x',[[1,1,0]]*2+[[1,0,0]]*2),('wrap y',[[1,0,0]]*2+[[1,1,0]]*2)]:
 print(name,rms(pred(np.reshape(q,(2,2,3)))-y))
def both(x): return np.repeat(np.reshape(x,(2,1,3)),2,axis=1)
def wraponly(x): return both([1,x[0],0,1,x[1],0])
for name,expand,p0 in [('wrap only',wraponly,[.5,.5]),('axis symmetric',both,[1,.5,0]*2),('four walls',lambda q:q.reshape(2,2,3),[1,.5,0]*4)]:
 for fold in [None,0,1,2]:
  tr=[j for j in range(3) if j!=fold]; fit=least_squares(lambda q:(pred(expand(q))[:,tr]-y[:,tr]).ravel(),p0,gtol=1e-12,max_nfev=350)
  print(name,fold,np.round(fit.x,6),'rms',rms(pred(expand(fit.x))-y))

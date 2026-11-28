exec(open('/tmp/mix_probe.py').read().split('configs=')[0])
r=[]
for name,g in [('base',K),('st',S),('sy',sy@sy.T)]:
 d=g.diagonal(); g=g/np.median(d)
 for kind,h in [('quadratic',g*g),('cubic',g**3),('smooth',g*g+.3*g**3),('normalized',(g/np.sqrt(g.diagonal()[:,None]*g.diagonal()[None,:]))**3)]:
  for a in [.03,.3,3]:
   for b in [0,3,30]:
    for lam in [.0001,.0003,.001,.003]:
     o=test(K+b*S+a*h,lam,False)
     if o[0]<.038:r.append((max(o),o[0],o[1:7].mean(),o[7],max(o[8:]),name,kind,a,b,lam))
for row in sorted(r)[:12]: print(row)

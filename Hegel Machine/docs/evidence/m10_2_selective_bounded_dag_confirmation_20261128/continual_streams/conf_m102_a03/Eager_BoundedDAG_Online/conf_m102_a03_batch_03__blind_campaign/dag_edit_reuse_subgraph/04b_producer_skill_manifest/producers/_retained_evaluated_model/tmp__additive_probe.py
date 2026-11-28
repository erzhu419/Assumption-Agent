exec(open('/tmp/panel_probe.py').read().split("for panel in")[0])
for ps in [(3,), (1,3), (1,2,3), (1,3,5), (1,2,3,4,5), (1,3,5,7)]:
 b=np.c_[np.ones(len(x)),*[x**p for p in ps]]
 for a in [0,.001,.01]:print(ps,a,cv(b,a))
for name,f in [('tanh',np.tanh),('sin',np.sin),('sinh',np.sinh),('atan',np.arctan),('cube',lambda u:u**3),('rat',lambda u:u**3/(1+u**2))]:
 for w in [.5,1,1.25,1.5,2,2.5,3]:
  b=np.c_[aff(x),f(w*x)]; print(name,w,cv(b))

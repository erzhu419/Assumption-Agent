exec(open('/tmp/panel_search.py').read().split('bases={}')[0])
from scipy.optimize import least_squares
v=np.roll(u,-1,axis=1)
def run(name, f, seeds):
 def err(k):
  F=np.c_[a,f(k)]; return (F@np.linalg.lstsq(F,y,rcond=1e-10)[0]-y).ravel()
 opts=[least_squares(err,k,max_nfev=150) for k in seeds]; o=min(opts,key=lambda o:o.cost); print(np.sqrt(np.mean(o.fun**2)),name,o.x)
for j in (1,3):
 v=np.roll(u,-j,axis=1)
 for powers in [(2,), (3,), (2,3),(2,3,4),(2,3,4,5),(3,5)]:
  terms=[u*v]+[u**i for i in powers]; run((j,powers),lambda k:terms[0]+sum(x*b for x,b in zip(k,terms[1:])),[np.zeros(len(powers))])
 for name,fn in [('sin',lambda z,k:np.sin(k*z)),('tanh',lambda z,k:np.tanh(k*z)),('rat3',lambda z,k:z/(1+k*z*z)),('rat2',lambda z,k:z*z/(1+k*z*z)),('odd_sat',lambda z,k:z/(1+k*abs(z))),('cos',lambda z,k:np.cos(k*z))]:
  for mode in (0,1,2,3):
   def f(k):
    if mode==0:return u*v+k[0]*fn(u,k[1])
    if mode==1:return np.c_[u*v,fn(u,k[0])]
    if mode==2:return u*v+k[0]*fn(u,k[1])+k[2]*u*u
    return u*v+k[0]*fn(u+k[2],k[1])
   seeds=[[a,b,c][:([2,1,3,3][mode])] for a in [-1,.1,1] for b in [.2,1,2.4] for c in [.0,.25]] if mode!=1 else [[.2],[1],[2.4],[4.]]
   run((j,name,mode),f,seeds)

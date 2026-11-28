exec(open('/tmp/probe07.py').read().split('for name,x')[0])
from scipy.linalg import qr
from scipy.optimize import minimize_scalar
spec=importlib.util.spec_from_file_location('sink',p+'node_01.py'); g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
u,v,q,pc=g._components(s,w); print('recovery',np.max(abs(u-f._inverse(raw,f._fit_panel_a(a['u'],a['v'])))))
acts={'x':lambda x:x,'sin':np.sin,'tanh':np.tanh,'sat':lambda x:x/(1+x*x)}
rows=[]
for i,(name,A) in enumerate((('v',v),('q',q),('u',u))):
 for name2,B in [('u',u),('v',v),('q',q)]:
  for k in [-3,-2,-1,-.5,0,.5,1,2,3]:
   S=A+k*B
   for name3,D in [('id',S),('tanh',np.tanh(2*S)),('sin',np.sin(2*S)),('sat',S/(1+1.25*S*S))]:
    for mode,F in [('pair',pair(D)),('poly',poly(D,2)[:,5:]),('cyclic',D*np.roll(D,-1,axis=1))]:
     for L,X in [('vq',aff(np.c_[v,q,F])),('v',aff(np.c_[v,F])),('q',aff(np.c_[q,F]))]:
      a0,a1=score(X); rows.append((*a1,*a0,name,name2,k,name3,mode,L))
 for name2,B in [('u',u),('v',v),('q',q)]:
  cross=(A[:,:,None]*B[:,None,:]).reshape(n,-1)
  for F in [cross,cross[:,[1,3,4,6,9,11,12,14]]]:
   for L,X in [('vq',aff(np.c_[v,q,F])),('vpq',aff(np.c_[v,q,pair(v),F])),('v',aff(np.c_[v,F]))]:
    a0,a1=score(X); rows.append((*a1,*a0,name,name2,'cross',F.shape[1],L))
for j in range(4): print('MIX',j,*sorted(rows,key=lambda r:r[j])[:7],sep='\n')
print('sorted residuals vs data')
F=aff(np.c_[v,q,pair(v)]); E=z-F@fit(F,z); print(np.c_[y,T[:,-1],E[:,0],E[:,-1],u].round(3))

exec(open('/tmp/fit_walls.py').read().split('for kind in')[0])
from scipy.sparse import diags

def periodic(d, adv, scale):
 k,v=s['kappa_'+d],s['vel_'+d]; h=s['L'+d]/(len(k)-1)
 left=(k+np.roll(k,1))/(2*h*h); right=(k+np.roll(k,-1))/(2*h*h)
 l=left+np.maximum(v,0)/h; r=right+np.maximum(-v,0)/h
 if not adv: l[[0,-1]]=left[[0,-1]]; r[[0,-1]]=right[[0,-1]]
 l[0]*=scale; r[-1]*=scale
 A=diags((-l[1:],l+r,-r[:-1]),(-1,0,1),format='lil'); A[0,-1]=-l[0]; A[-1,0]=-r[-1]
 return A.tocsc()
for d in 'xy':
 k,v=s['kappa_'+d],s['vel_'+d]; h=s['L'+d]/(len(k)-1)
 print(d,'k endpoints',k[[0,1,-2,-1]],'v endpoints',v[[0,-1]],'h',h)
 for adv in [False, True]:
  A=periodic(d,adv,1).toarray(); den=.5*(k[0]+k[-1])/h**2
  print(adv,'m and c',-A[0,1]*h*h/k[0],-A[-1,-2]*h*h/k[-1],-A[0,-1]/den,-A[-1,0]/den)
for a in [False,True]:
 for b in [False,True]:
  e=arc._responses(s,[periodic('x',a,1),periodic('y',b,1)])[0]@W-Y
  print(a,b,'pulse rms',np.sqrt(np.mean(e**2,axis=0)),'peak',abs(e).max())

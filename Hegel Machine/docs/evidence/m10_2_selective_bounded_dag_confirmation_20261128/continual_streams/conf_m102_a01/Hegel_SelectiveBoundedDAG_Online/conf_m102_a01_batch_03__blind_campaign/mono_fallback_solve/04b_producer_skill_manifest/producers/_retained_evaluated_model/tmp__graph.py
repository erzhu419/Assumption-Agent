from explore import *
from scipy.optimize import minimize_scalar
raw=c['u']
def panel(g,verbose=False):
 out=[]
 for key,func,limits in [('v1',np.tanh,(.1,5)),('v1',np.sin,(.1,4)),('v2',np.cos,(.1,4))]:
  def f(a,k):return np.c_[aff(a),func(k*a)]
  def err(k):
   F=f(raw,k); return np.mean((F@fit(F,c[key])-c[key])**2)
  t=minimize_scalar(err,bounds=limits,method='bounded'); F=f(raw,t.x); q=fit(F,c[key]);
  if verbose: print(key,func.__name__,t.x,t.fun**.5, '\n', q.round(4))
  out.append((t.fun,key,f(g,t.x)@q))
 return min(out[:2],key=lambda p:p[0])[2],out[2][2]
def fw(g):return g@B[1:]+B[0]
def bw(g):return (g-B[0])@np.linalg.inv(B[1:]);
if __name__=='__main__':
 panel(raw,True); print('B svd',np.linalg.svd(B[1:])[1]);
 G={'x':x,'z':z,'h':h,'Bx':fw(x),'Bz':fw(z),'ix':bw(x)}
 for k,g in list(G.items()):
  a,b=panel(g); G.update({k+'c1':a,k+'c2':b,k+'bc1':fw(a),k+'bc2':fw(b),k+'ic1':bw(a),k+'ic2':bw(b),k+'c1+c2':a+b,k+'c1*c2':a*b})
 def compare(F):
  q=fit(F[:52],R[:52]); return np.r_[np.mean(abs(F[52:]@q-R[52:]),0),np.std(F[:52]@q-R[:52],0)]
 opts=[]
 for k,g in G.items():
  for tag,F in [('lin',aff(g)),('quad',poly(g,2)),('all',np.c_[aff(g),x])]:opts.append((k,tag,compare(F)))
 for k in ['x','z','Bx','Bz','ix','h']:
  a,b=G[k+'c1'],G[k+'c2']; g=G[k]
  for tag,F in [('ab',np.c_[aff(a),b]),('gab',np.c_[aff(a),b,g]),('abprod',np.c_[aff(a),b,a*b]),('abprod2',np.c_[aff(a),b,(a[:,:,None]*b[:,None,:]).reshape(len(a),16)]),('q+b',np.c_[poly(a,2),b]),('q+a',np.c_[poly(b,2),a])]:opts.append((k,tag,compare(F)))
 for j in range(3):
  print('BEST',j); print(*sorted(opts,key=lambda a:a[2][j])[:15],sep='\n')

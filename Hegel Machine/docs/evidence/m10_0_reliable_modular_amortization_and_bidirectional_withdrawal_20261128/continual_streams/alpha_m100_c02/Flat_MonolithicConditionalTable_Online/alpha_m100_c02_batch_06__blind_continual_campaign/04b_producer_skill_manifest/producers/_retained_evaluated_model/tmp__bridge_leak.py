exec(open('/tmp/bridge_grammar.py').read().split('for k,l,inter in')[0])
from scipy.optimize import minimize_scalar
for mode in ('poly3','poly4','poly5','tanh','atan','rat','sin','asinh','rat_even','tanh_even','sin_even'):
 def basis(a,b,w):
  cols=[np.ones(len(a)),a,np.sin(w[0]*a),b]
  t=w[-1]*b
  if mode.startswith('poly'): cols.extend(b**i for i in range(2,1+int(mode[-1])))
  else:
   cols.append({'tanh':np.tanh(t),'atan':np.arctan(t),'rat':b/(1+t*t),'sin':np.sin(t),'asinh':np.arcsinh(t),'rat_even':b*b/(1+t*t),'tanh_even':np.tanh(t)**2,'sin_even':1-np.cos(t)}[mode]); cols.append(b*b) if not mode.endswith('even') else None
  return np.column_stack(cols)
 es=[]; fs=[]
 for f in range(5):
  test=good&mixed&(fold==f); data=[setup(j,good&~test,sg) for j in range(4)]
  def fit(w,coef=False):
   out=[]
   for a,b,y in data:
    A=basis(a,b,w); q=np.linalg.lstsq(A,y,rcond=None)[0]; out.append(q if coef else A@q-y)
   return out if coef else np.concatenate(out)
  opts=[least_squares(fit,[2.5,w],bounds=([2,.1],[3,5]),max_nfev=70) for w in (.5,1.5,3)]
  w=min(opts,key=lambda x:x.cost).x; q=fit(w,True); fs.append(w)
  es.extend(basis(L[test,j],L[test,(j+2)%4],w)@q[j]-R[test,j] for j in range(4))
 report(mode,es); print('shape',np.round(fs,3).tolist(),'coef',np.round(q,4).tolist())

exec(open('/tmp/hegel_response_probe.py').read().split('for y in [Y,T]:')[0])
from scipy.optimize import minimize_scalar
D=np.load('/tmp/hegel_b_probes.npz');
for label,z in [('aff',w),('rational',W),('tanh',D['intanh1'])]:
 for y in [Y,T]:
  rows=[]
  for typ,fn in [('atan',np.arctan),('inv3',lambda q:np.sinh(np.arcsinh(q)/3)),('asinh',np.arcsinh),('tanh',np.tanh),('rat',lambda q:q/(1+q*q)),('cubic',lambda q:q**3)]:
   for g in [.35,.7,1.25,2,3,5,8]:
    p=fn(z*g)
    for deg in [1,2]:
     for extra in [[],[C1],[C2],[C1,C2],[u]]:
      F=np.column_stack([pol(p,deg),z]+extra);e=ev(F,y);rows.append((e[0],e[1],typ,g,deg,len(extra), 'C1' if extra and extra[0] is C1 else 'other',e[2]))
  print(label,y.shape,*sorted(rows,key=lambda z:z[0])[:5],sep='\n')
print('cross y and trajectory',np.corrcoef(Y,T[:,-1])); print('telemetry',s['telemetry_vec']); print('schema',list(s),[(k,list(v) if isinstance(v,dict) else np.shape(v)) for k,v in s.items()])

exec(open('/tmp/probe_dag.py').read().split("print('times'")[0])
from scipy.optimize import least_squares
from scipy.special import erf
links={'lin':lambda w:w,'tanh':np.tanh,'sin':np.sin,'soft':lambda w:w/np.sqrt(1+w*w),'atan':np.arctan,'erf':erf,'square':lambda w:w*w,'cubic':lambda w:w**3}
for k in ['panel_b','panel_c']:
 p=s[k]; F=a(p['u']); print(k)
 for key in sorted(set(p)-{'u'}):
  V=p[key]; print(key,'range',V.min(0),V.max(0),'row[0:3]',V[:3]);
  for name,link in links.items():
   for mode in [0,1,2]:
    errs=[]; pars=[]
    for v in V.T:
     w=np.linalg.lstsq(F,v,rcond=None)[0]; init=np.r_[w,1.,0.]
     def f(b):return (b[5] if mode else 1)*link(F@b[:5])+(b[6] if mode==2 else 0)-v
     r=least_squares(f,init,max_nfev=300); errs.append(np.sqrt(np.mean(r.fun**2))); pars.append(r.x[5:])
    print(name,mode,np.round(errs,6),np.round(pars,3) if max(errs)<.012 else '')
 print('other panel relation')
 if k=='panel_c':
  for degree in [1,2,3]:
   for inds in [('u','v1'),('v1',)]:
    Q=np.column_stack([p[l] for l in inds]); G=np.column_stack([np.ones(len(Q))]+[np.prod(Q[:,j],1) for d in range(1,degree+1) for j in comb(range(Q.shape[1]),d)]); H=G@np.linalg.pinv(G); R=(np.eye(len(Q))-H)@p['v2']; print(degree,inds,np.sqrt(np.mean(R**2,0)))

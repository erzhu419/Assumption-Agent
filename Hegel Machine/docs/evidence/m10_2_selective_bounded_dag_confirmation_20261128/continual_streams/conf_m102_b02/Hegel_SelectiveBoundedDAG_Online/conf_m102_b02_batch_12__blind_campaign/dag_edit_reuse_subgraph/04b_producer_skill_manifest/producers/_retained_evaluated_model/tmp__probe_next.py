from probe import *
from itertools import combinations_with_replacement
from scipy.optimize import root
ua,va=A['u'],A['v']; idx=[j for k in range(1,4) for j in combinations_with_replacement(range(4),k)]
def des(q,idx): return np.c_[np.ones(len(q)),*[np.prod(q[:,j],axis=1) for j in idx]]
keep=np.max(abs(va),axis=1)<4
wa=np.linalg.lstsq(des(ua,idx)[keep],va[keep],rcond=None)[0]
print('A terms',[(j,np.round(row,5).tolist()) for j,row in zip([()]+idx,wa) if np.linalg.norm(row)>.0002],flush=True)
idx=[j for j,row in zip(idx,wa[1:]) if np.linalg.norm(row)>.0002];wa=np.linalg.lstsq(des(ua,idx)[keep],va[keep],rcond=None)[0]
print('A sparse residual',np.sqrt(np.mean((des(ua,idx)[keep]@wa-va[keep])**2)),flush=True)
xx=np.array([root(lambda q: (des(q[None],idx)@wa)[0]-obs,start).x for obs,start in zip(x0,x)]);print('A inverse error',abs(des(xx,idx)@wa-x0).max(), 'range',xx.min(),xx.max(),flush=True)
D=lambda q,*cols:np.column_stack((np.ones(len(q)),*cols))
_,_,g=fit(u,v2,'rat2even')
maps={'ratodd':lambda q,p:q/(1+np.exp(p[0])*q*q),'arcsinh':lambda q,p:np.arcsinh(np.exp(p[0])*q),'tanh':lambda q,p:np.tanh(np.exp(p[0])*q),'erf3':lambda q,p:erf(np.exp(p[0])*q+p[1]*q**3),'ratabs':lambda q,p:q*abs(q)/(1+np.exp(p[0])*abs(q)),'rata':lambda q,p:q/(1+np.exp(p[0])*abs(q))}
Bs={'arch':fit(ub,vb,'erf3')[2](xx)}; Cs={'arch':fit(u,v,'rat3')[2], 'v2':g}
for key,Q,V in [('B',ub,vb),('C',u,v)]:
 opts=[]
 for name,fn in maps.items():
  for extra in ['odd','even','companion'] if key=='C' else ['odd','even']:
   def d(q,p,fn=fn,extra=extra):return D(q,fn(q,p),q,q**3,*([q*q] if extra=='even' else [g(q)[:,1:]] if extra=='companion' else []))
   starts=[[.6,4],[.4,-.27]] if name=='erf3' else [[0],[-1],[1]]
   e,p,predict=vp(Q,V,d,starts); label=name+'_'+extra; print(key,label,round(e,8),p.round(5),flush=True);opts.append((e,label,predict))
 for e,label,predict in sorted(opts)[:6]:
  if key=='B':Bs[label]=predict(xx)
  else:Cs[label]=predict
for bn,bb in Bs.items():
 for cn,cf in Cs.items():
  sc=np.array(score(cf(bb),xx)); print('score',bn,cn,sc[0],sc.max(0),flush=True)

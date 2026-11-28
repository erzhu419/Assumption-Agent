exec(open('/tmp/operator01.py').read().split('for base in')[0])
from scipy.linalg import qr
D={'1':ones}; F={};
for prefix,a in [('x',x),('b',b)]:
 for j,w in enumerate(a.T):
  k=f'{prefix}{j}'; D[k]=w; F[k]=w
  for power in (2,3,4,5): D[f'{k}^{power}']=w**power
  D[f'abs({k})']=abs(w); D[f'tanh({k})']=np.tanh(w); D[f'sin({k})']=np.sin(w); D[f'cos({k})']=np.cos(w)
  for scale in (.25,1.,4.):
   D[f'odd({k},{scale})']=w/(1+scale*w*w); D[f'even({k},{scale})']=w*w/(1+scale*w*w)
for (k,w),(j,g) in itertools.combinations(F.items(),2):
 D[k+'*'+j]=w*g
 for s in (1,-1):
  h=w+s*g; name=f'({k}{s:+d}{j})'; D[f'odd{name}']=h**3/(1+h*h); D[f'even{name}']=h*h/(1+h*h)
for k,w in F.items():
 for j,g in F.items():
  if j!=k: D[f'{k}^2*{j}']=w*w*g; D[f'{k}/(1+{j}^2)']=w/(1+g*g); D[f'{k}^2/(1+{j}^2)']=w*w/(1+g*g)
for (k,w),(j,g),(m,h) in itertools.combinations(F.items(),3): D[k+'*'+j+'*'+m]=w*g*h
for j,w in enumerate(x.T):
 for k,g in enumerate(x.T):
  D[f'x{j}*r{k}']=w*rational(g,p)
  D[f'evenr{j}/{k}']=w*w/(1+p*g*g)
names=list(D); M=np.column_stack(list(D.values())); ids={k:i for i,k in enumerate(names)}
base=[ids[k] for k in ['1']+[f'x{j}' for j in range(4)]+[f'b{j}' for j in range(4)]]

def search(tr,base,depth=12,beam=12,allowed=None):
 Y=y[tr]; A=M[tr]; n=len(tr); candidates=np.arange(len(names)) if allowed is None else np.array(allowed); active=[tuple(base)]; solutions=[]
 for i in range(depth+1):
  nxt={}; best=None
  for use in active:
   U=A[:,use]; Q=np.linalg.qr(U)[0]; c=np.linalg.lstsq(U,Y,rcond=1e-9)[0]; e=Y-U@c; h=(Q*Q).sum(1); loo=np.mean(abs(e/(1-h)))
   if best is None or loo<best[0]: best=(loo,use,c,np.mean(abs(e)))
   if i==depth: continue
   V=A[:,candidates]; V=V-Q@(Q.T@V); sq=(V*V).sum(0); good=sq>1e-7; proj=(e@V)/np.maximum(sq,1e-30); en=e[:,None]-V*proj; scores=np.mean(abs(en/np.maximum(1-h[:,None]-(V*V)/np.maximum(sq,1e-30),1e-7)),axis=0); scores[~good]=np.inf
   for j in np.argsort(scores)[:beam]:
    if np.isfinite(scores[j]): nxt[tuple(sorted((*use,candidates[j])))]=scores[j]
  if best is None: break
  solutions.append(best)
  active=sorted(nxt,key=nxt.get)[:beam]
 return solutions
splits=[np.arange(52,72),*np.array_split(np.arange(len(x)),4)]
for key,start in [('b',base[:1]+base[5:]),('xb',base)]:
 counts=[]; selected=[]
 for i,te in enumerate(splits):
  tr=np.setdiff1d(np.arange(len(x)),te); r=search(tr,start); rows=[]
  for loo,use,c,train in r:
   held=np.mean(abs(M[te][:,use]@c-y[te])); rows.append(held)
  best=min(r,key=lambda q:q[0]); loo,use,c,train=best; chosen=np.mean(abs(M[te][:,use]@c-y[te])); print(key,i,'CV path',np.round(rows,4),'selected',round(chosen,5),'LOO',round(loo,4),[(names[j],round(co,3)) for j,co in zip(use,c)],flush=True); counts.append(rows); selected.append(chosen)
 print(key,'worst by depth',np.max(counts,axis=0),'nested worst',max(selected),flush=True)

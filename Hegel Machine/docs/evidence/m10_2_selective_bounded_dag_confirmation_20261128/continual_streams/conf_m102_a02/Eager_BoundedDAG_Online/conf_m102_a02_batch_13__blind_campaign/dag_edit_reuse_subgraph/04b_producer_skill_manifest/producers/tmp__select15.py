import itertools as it,pickle,numpy as np
from final15 import coordinates,endpoint,fit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x'];y=s['train']['y']; tr=s['train']['traj']; n=len(x); ids=np.arange(n)
a=lambda x:np.c_[np.ones(len(x)),x]
u,g=coordinates(s,{'a':a(x),'a_test':a(s['test']['x'])}); u,g=u[:n],g[:n]
folds=[ids[-20:]]+[v for seed in [9,42,112] for v in np.array_split(np.random.default_rng(seed).permutation(n),4)]
# Add a shifted operating regime, rather than only random row splits.
folds += [np.argsort(u[:,0]-u[:,2])[:12],np.argsort(u[:,0]-u[:,2])[-12:]]
fsets=[(np.setdiff1d(ids,v),v) for v in folds]
res=[]
for (b,p),num,f,sft in it.product([('cross',0),('quad',0),('quad',.03),('quad',.3),('quad',3.)],['one','pair','all','const','cross'],[.15,.22,.35,.5,.8,0],[.75,1.,1.25]):
 spec=(b,num,p,f,sft,False); es=[]
 for t,v in fsets:
  o=np.r_[t,v]; X,c=endpoint(u[o],g[o],y[t],spec);es.append(np.abs(X[len(t):]@c-y[v]).mean())
 res.append((spec,np.array(es)))
for num in ['none','one','pair','const','cross']:
 for f in [.22,.4,.65,0]:
  spec=('cross',num,0.,f,1.,True); es=[]
  for t,v in fsets:
   o=np.r_[t,v]; X,c=endpoint(u[o],g[o],y[t],spec);es.append(np.abs(X[len(t):]@c-y[v]).mean())
  res.append((spec,np.array(es)))
passing=[(k,e) for k,e in res if e[0] <= .0375]
score=lambda r: .40*r[1][1:13].max()+.40*r[1][1:13].mean()+.20*r[1][13:].max()
ranked=sorted(passing or res,key=score if passing else lambda r:r[1][0]);
for k,e in ranked[:7]:print(k, 'errors',e.round(5),'score',score((k,e)))
print('trajectory slices', [round(np.abs(a(g[v])@fit(a(g[t]),tr[t])-tr[v]).mean(),6) for t,v in fsets]); print('official-pass candidates',len(passing))
with open('/holdout_task/node_01.py','w') as dest:
 dest.write('"""Robust inverse calibration and independent scalar and temporal readouts."""\n'+open('/tmp/final15.py').read()+'\n\nSPEC = '+repr(ranked[0][0])+'\n')

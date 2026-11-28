exec(open('/tmp/panel_probe.py').read().split("for i,j in [(0,0)")[0])
from itertools import combinations
from functools import partial
frozen=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')
def panel(w,key):
 params={'v1':('soft',.836),'v2':('atan',1.078)}
 kind,k=params[key]; g=lib[kind]
 def design(a):return bias(np.c_[a,g(a,k),a*np.roll(a,1,axis=1)])
 return design(w)@ls(design(u),p[key])
a,b=panel(z,'v1'),panel(z,'v2'); R=ls(bias(s['panel_b']['u']),s['panel_b']['v']); rot=bias(z)@R
pairs=[('AB',a,b),('ZA',z,a),('ZB',z,b),('ZR',z,rot),('RA',rot,a),('RB',rot,b)]
def quad(w): return np.c_[w,*[w[:,i]*w[:,j] for i in range(w.shape[1]) for j in range(i,w.shape[1])]]
def pred(f,i,j,r):
 fi,fj=f[i],f[j]; mu=fi.mean(0); sd=np.maximum(fi.std(0),1e-8); fi=(fi-mu)/sd; fj=(fj-mu)/sd; y=yy[i]; ym=y.mean(0); return fj@(ls(fi,y-ym) if r==0 else np.linalg.solve(fi.T@fi+r*np.eye(fi.shape[1]),fi.T@(y-ym)))+ym
sp=[]
for i,j in cv:
 ss=dict(s,train={k:s['train'][k][i] for k in ('x','y','traj')},test={'x':x[j]}); o=frozen['run_node']({'__input__':ss}); sp.append(np.c_[o['y_hat'],o['state_pred']])
def evals(ps):
 es=[]
 for (i,j),pr in zip(cv,ps):
  e=abs(pr-yy[j]); es.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.array(np.r_[es[0],np.mean(es[1:],0),np.max(es[1:],0)])
print('sparse',evals(sp)); results=[]
for name,a,b in pairs:
 w=np.c_[a,b]
 for kind,f in [('old',np.c_[w,(a*b).sum(1),a[:,0]*b[:,1]]),('within',np.c_[quad(a),quad(b),a*b]),('quad',quad(w)),('raw',w),('cross',np.c_[w,*[a[:,i]*b[:,j] for i in range(4) for j in range(4)]])]:
  for r in [.001,.03,.1,.3,1]:
   pp=[pred(f,i,j,r) for i,j in cv]; mix=[v+np.array([.5]+[.65]*8)*(p-v) for p,v in zip(pp,sp)]
   for mode,pr in [('direct',pp),('blend',mix)]:
    e=evals(pr); results.append((.3*max(e[:2])+.7*max(e[4:]),name,kind,r,mode,np.round(e,5)))
for row in sorted(results,key=lambda x:x[0])[:22]:print(row)

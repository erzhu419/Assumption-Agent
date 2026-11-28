from features import *
from itertools import product,combinations
from scipy.special import erf
p=s['panel_b']; q=s['panel_c']
B=lambda x:D(np.sign(x)*np.maximum(abs(x)-.35,0)**1.5)
C=lambda x:D(x*np.roll(x,-1,axis=1))
E=lambda x:D(np.sqrt(1+.924227*x*x)+.831839*x)
WB=fit(B(p['u']),p['v']); WC=fit(C(q['u']),q['v1']); WE=fit(E(q['u']),q['v2']); ops=[lambda w:B(w)@WB,lambda w:C(w)@WC,lambda w:E(w)@WE]
P=lambda w:w*np.roll(w,-1,axis=1)
cb=ops[1](b); eb=ops[2](b); print('panel weights',WB.round(3),WC.round(3),WE.round(3)); print('traj, y linear',[np.std(T[:52]-D(np.c_[z,b,v])[:52]@fit(D(np.c_[z,b,v])[:52],T[:52]),axis=0).round(4) for v in (cb,eb,np.c_[cb,eb])]);
fs={'z':z,'b':b,'c':c,'e':e,'cb':cb,'eb':eb,'p':P(b)}
for a,u in list(fs.items()):
 for k,h in enumerate(ops):
  for r in (1.,1.25):fs[f'{k}({r}{a})']=h(r*u)
for a,u in [('z',z),('cb',cb),('eb',eb),('e',e)]:
 for k in (.176,.22,.275,.5,1):
  for sign in [-1,1]:
   v=b+sign*k*u;fs[f'b{sign:+d}*{k}{a}']=v
   for h,o in enumerate(ops):fs[f'{h}(b{sign:+d}*{k}{a})']=o(v)
act={'id':lambda v:v,'sq':np.square,'cube':lambda v:v**3,'abs':abs,'sin':np.sin,'tanh':np.tanh,'asinh':np.arcsinh,'erf':erf,'rat':lambda v:v/(1+v*v),'sqrt':lambda v:np.sqrt(1+v*v),'softplus':lambda v:np.logaddexp(0,v),'exp':lambda v:np.exp(np.clip(v,-30,12))}
base=[D(np.c_[z,b]),D(np.c_[z,b,cb]),D(np.c_[z,b,cb,eb]),D(np.c_[z,b,cb,e]),D(np.c_[z,b,cb,P(z),z*z,b*b]),D(np.c_[z,b,cb,eb,g@fit(g[:52],y[:52])])]; S=[]
for nm,v in fs.items():
 for fn,f in act.items():
  for sc in ((1.,) if fn in ['id','sq','cube','abs'] else (.5,1.25,2.5,4.)):
   w=f(sc*v); w=(w-w[:52].mean(0))/np.maximum(w[:52].std(0),1e-6)
   for k,X in enumerate(base):
    A=np.c_[X,w]; scores=[]
    for i,j in splits:
     coef=np.linalg.solve(A[i].T@A[i]+.001*np.eye(A.shape[1]),A[i].T@T[i]); R=abs(A[j]@coef-T[j]);scores.append([R[:,0].mean(), R[:,1:].mean()]);
    S.append((np.array(scores),nm,fn,sc,k))
for k in (0,1):
 for sort in (lambda r:r[0][:,k].mean(),lambda r:r[0][0,k]):
  for r in sorted(S,key=sort)[:12]:print(k,r[0][:,k].round(5),r[1:]);

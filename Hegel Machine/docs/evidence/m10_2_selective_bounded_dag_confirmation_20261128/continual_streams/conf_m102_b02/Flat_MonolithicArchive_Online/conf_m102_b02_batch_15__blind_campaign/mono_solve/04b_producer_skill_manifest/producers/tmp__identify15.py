import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda X,y: np.linalg.lstsq(X,y,rcond=1e-9)[0]
def des(*x):return np.c_[np.ones(len(x[0])),*x]
F={'lin':lambda x,q:x,'sq':lambda x,q:x*x,'cube':lambda x,q:x**3,'sin':lambda x,q:np.sin(q*x),'cos':lambda x,q:np.cos(q*x),'tanh':lambda x,q:np.tanh(q*x),'asinh':lambda x,q:np.arcsinh(q*x),'rat':lambda x,q:x/(1+(q*x)**2),'rad':lambda x,q:1/(1+(q*x)**2),'exp':lambda x,q:np.exp(-(q*x)**2),'adj':lambda x,q:x*np.roll(x,-1,axis=1),'opp':lambda x,q:(x*np.roll(x,2,axis=1))[:,:2],'abs':lambda x,q:np.abs(x)}
fixed={'lin','sq','cube','adj','opp','abs'}
if __name__=='__main__':
 u,v=s['panel_a'].values(); d=des(u); err=np.linalg.norm(d@ls(d,v)-v,axis=1); keep=err<4; u,v=u[keep],v[keep];
 for j in range(4):
  out=[]
  for a,b in itertools.combinations_with_replacement(F,2):
   if a in ('adj','opp') or b in ('adj','opp'):continue
   for k in range(4):
    for h in range(4):
     def mat(q):return des(u,F[a](u[:,k],q[0]),F[b](u[:,h],q[1]))
     def fun(q):
      d=mat(q); return d@ls(d,v[:,j])-v[:,j]
     vals=[least_squares(fun,[x,x],bounds=(.12,9),max_nfev=27) for x in (1.5,3.8)]
     p=min(vals,key=lambda x:x.cost);out.append((np.std(p.fun),a,k,b,h,p.x,ls(mat(p.x),v[:,j])))
  print('A',j,sorted(out,key=lambda x:x[0])[:3],flush=True)
 for panel,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u=s[panel]['u'];v=s[panel][key];out=[]
  for cnt in [1,2,3]:
   for fs in itertools.combinations(F,cnt):
    if cnt==3 and 'lin' not in fs: continue
    act=[i for i,f in enumerate(fs) if f not in fixed]; m=len(act)
    def mat(q):
     p=np.ones(cnt);p[act]=q;return des(*(F[f](u,p[i]) for i,f in enumerate(fs)))
    def fun(q):
     d=mat(q);return (d@ls(d,v)-v).ravel()
    if m:
     basis=[.7,1.6,2.6,3.7,5.5]; vals=[least_squares(fun,q,bounds=(.12,9),max_nfev=60) for q in itertools.product(basis,repeat=m)]; p=min(vals,key=lambda x:x.cost).x
    else:p=[]
    out.append((np.std(fun(p)),fs,p))
   print(panel,key,cnt,sorted([x for x in out if len(x[1])==cnt],key=lambda x:x[0])[:7],flush=True)

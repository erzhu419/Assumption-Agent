exec(open('/tmp/panel_search.py').read().split('C={')[0])
from scipy.optimize import least_squares
from itertools import product
rng=np.random.default_rng(45); ix=rng.permutation(len(y)); j=ix[:18]; i=ix[18:]; Y=np.c_[y,t];
print('studies',len(pickle.load(open('/holdout_task/public_batch.pkl','rb'))),'x bounds',x.min(0),x.max(0),'z bounds',z.min(0),z.max(0),'test bounds',s['test']['x'].min(0),s['test']['x'].max(0)); print('sv',np.linalg.svd(t,compute_uv=False));
def err(F):
 r=F[j]@ls(F[i],Y[i])-Y[j]; return abs(r[:,0]).mean(),abs(r[:,1:]).mean()
ops2={'z':z,'x':x,'Bz':z@B+h,'invz':(z-h)@P}; out=[]; nonlinear=[]
for nm,w in ops2.items():
 c,d=[f(w) for f in funcs]; dp=(c*d).sum(1); cr=(c[:,:,None]*d[:,None,:]).reshape(len(c),-1); base=ones(np.c_[c,d,dp]);
 for un,U in [('w',w),('c',c),('d',d),('c+d',c+d),('c-d',c-d),('w+c',w+c),('w+d',w+d),('cd',c*d)]:
  for g,G in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('rat',lambda a:a/(1+a*a)),('evenrat',lambda a:1/(1+a*a)),('soft',lambda a:a/(1+abs(a))),('abs',abs),('exp',lambda a:np.exp(-a*a))]:
   for k in [.5,1.25,2.15,3.5,5]:
    V=G(k*U); out.append((*err(np.c_[base,V]),nm,un,g,k,'plain')); out.append((*err(np.c_[base,cr,V]),nm,un,g,k,'cross'))
 for fn,V in [('branches',np.c_[c,d]),('branch_dot',np.c_[c,d,dp]),('cross',np.c_[c,d,cr]),('raw',w),('combined',np.c_[w,c,d,dp])]:
  F=ones(V); F=F/np.maximum(F.std(0),1e-8) if False else F/np.sqrt(np.mean(F**2,axis=0));
  for g,G in [('tanh',np.tanh),('sin',np.sin),('soft',lambda a:a/(1+abs(a))),('atan',np.arctan),('rat',lambda a:a/(1+a*a)),('asinh',np.arcsinh)]:
   for mode in [0,1,2]:
    # Squashed full score, possibly added to a linear/cross component.
    A0=ones(w) if mode==1 else (base if mode==2 else np.ones((len(w),1))); Ti=t[:,-1];
    for tr,v in [('y',y),('t',Ti)]:
     amp=max(np.std(v),.5); lin=ls(F[i],v[i]);
     def pred(p):
      non=G(F@p[:-1]); return A0@ls(A0[i],v[i]-p[-1]*non[i])+p[-1]*non
     def loss(p):return (pred(p)-v)[i]
     if fn=='cross' and mode>0:continue
     fits=[least_squares(loss,np.r_[lin/a,a],max_nfev=65) for a in [amp,amp*2,amp*4]]; p=min(fits,key=lambda f:f.cost); nonlinear.append((tr,np.mean(abs((pred(p.x)-v)[j])),np.sqrt(np.mean(p.fun**2)),nm,fn,g,mode))
print('fixed y',sorted(out)[:12]); print('fixed t',sorted(out,key=lambda p:p[1])[:12]);
for tr in ('y','t'):
 print('index',tr,sorted([q for q in nonlinear if q[0]==tr],key=lambda p:p[1])[:16]); print('index training',tr,sorted([q for q in nonlinear if q[0]==tr],key=lambda p:p[2])[:6]);

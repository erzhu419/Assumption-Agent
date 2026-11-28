exec(open('/tmp/graph.py').read().split("print('B'")[0])
from scipy.optimize import least_squares
tr=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; n=52
print('TELEMETRY AND TIMES',s['telemetry_vec'],t,'calibration',coef,flush=True)
P=ops['P'](z); Q=ops['Q'](z); B=ops['B'](z)
def soft(a):return np.logaddexp(0,a)
def pp(a): return (a[:,:,None]*a[:,None,:])[:,np.triu_indices(4)[0],np.triu_indices(4)[1]]
def designs(k):
 sp=soft(k*P); sq=soft(k*Q); R=np.sum(P*Q,1)[:,None];
 return {'P':P,'Q':Q,'softP':sp,'softQ':sq,'softP+Q':np.c_[sp,Q],'P+Q':np.c_[P,Q],'P+Q+dot':np.c_[P,Q,R],'softP+Q+dot':np.c_[sp,Q,R],'softP+Q+prod':np.c_[sp,Q,P*Q],'softP+Q+softprod':np.c_[sp,Q,sp*Q],'softP+P':np.c_[sp,P],'Q+cross':np.c_[Q,pp(Q)],'P+cross':np.c_[P,pp(P)],'P Q rat':np.c_[P,Q,P/(1+k*Q*Q)],'P Q soft projection':np.c_[P,Q,soft(k*(P+Q))],'P Q soft diff':np.c_[P,Q,soft(k*(P-Q))],'P Q soft dot':np.c_[P,Q,soft(k*R)]}
H=-np.expm1(-t[:,None]*np.array([1.8,.4,5.8])); modes=ls(H,tr.T).T
for j,v in enumerate([y,*modes.T]):
 r=[]
 for k in [.2,.7,1.25,2.4,4,7,12]:
  for a,f in designs(k).items():
   f=D(f); e=v-f@ls(f[:n],v[:n]); lev=np.sum(f[:n]*np.linalg.pinv(f[:n]).T,1); r.append((np.mean(abs(e[:n]/(1-lev))),np.mean(abs(e[n:])),k,a))
 print('MOD',j,sorted(r)[:6],flush=True)
# Profile unknown affine ridge nonlinearities, both signs and multiplicative interactions.
from scipy.special import expit
rng=np.random.default_rng(7)
for j,v in enumerate([y,modes[:,0],modes[:,1]]):
 rr=[]
 for key,F,W in [('PQ',np.c_[P,Q],np.c_[P,Q]),('P',P,np.c_[P,Q]),('Q',Q,np.c_[P,Q]),('z',z,np.c_[P,Q]),('zPQ',np.c_[z,P,Q],np.c_[P,Q]),('B',B,np.c_[P,Q])]:
  for nm,fun in [('soft',soft),('sin',np.sin),('tanh',np.tanh),('rat',lambda a:1/(1+a*a)),('gate',expit)]:
   for variant in [0,1]:
    W0=D(W); d=F.shape[1]+1
    def pred(a):
     g=fun(D(F)@a); return np.c_[W0,g] if variant==0 else np.c_[W0,W0*g[:,None]]
    def obj(a):
     f=pred(a)[:n]; return f@ls(f,v[:n])-v[:n]
    for _ in range(4):
     r=least_squares(obj,rng.normal(0,.75,d),max_nfev=65); f=pred(r.x); e=f@ls(f[:n],v[:n])-v; rr.append((np.mean(abs(e[:n])),np.mean(abs(e[n:])),key,nm,variant,r.x.round(2)))
 print('RIDGE',j,*sorted(rr,key=lambda a:a[0])[:5],sep='\n',flush=True)

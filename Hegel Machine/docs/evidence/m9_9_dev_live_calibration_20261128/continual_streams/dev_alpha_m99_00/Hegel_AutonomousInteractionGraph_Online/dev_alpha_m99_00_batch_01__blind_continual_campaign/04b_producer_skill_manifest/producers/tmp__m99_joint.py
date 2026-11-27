exec(open('/tmp/m99_probe.py').read().split('for d,c in')[0])
y,dy=obs(di); Y=np.c_[y,dy]; p,h,d=cd; D=d@th[2:]; dt=np.asarray(dt)
def gap(u):
 z=np.zeros((len(h),2))
 for j,s in enumerate(dt): z+=-np.expm1(-s/u)*(p[:,j,None]-z)
 return h[:,None]-z

def fit(A,idx=None,weight=3.,start=(.5,2.5),print_co=False):
 if idx is None: idx=np.arange(len(h))
 w=np.r_[np.ones(len(t)),weight]
 def basis(q):
  u=np.exp(q); return np.r_[np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1][None,:]]
 def design(q,idx):
  B=basis(q); return np.column_stack([(a[idx,None,:]*B[None,:,i,None]).reshape(len(idx)*len(w),-1) for i,a in enumerate(A)])
 def decomp(q):
  F=design(q,idx); W=np.tile(w,len(idx)); return F, np.linalg.lstsq(F*W[:,None],Y[idx].ravel()*W,rcond=None)[0]
 def residual(q):
  F,c=decomp(q); return ((F@c-Y[idx].ravel()).reshape(-1,len(w))*w).ravel()
 sol=least_squares(residual,np.log(start),bounds=(np.log([.15,1]),np.log([.99,15])),max_nfev=80)
 F,c=decomp(sol.x); R=(design(sol.x,np.arange(len(h)))@c).reshape(Y.shape)-Y
 if print_co: print('taus',np.exp(sol.x).round(5),'co',np.round(c,5))
 return R

def stats(r): return np.r_[np.mean(r[:,:-1]**2)**.5,np.mean(r[:,-1]**2)**.5]
def fmt(r): return np.round(stats(r),5)
N=len(h); one=np.ones(N); hist=h[:,None]-gap(tau); x=d[:,1:]; lab=di['lab_id']; mean=p@dt/sum(dt); groups=[np.flatnonzero(lab==k) for k in np.unique(lab)]
F=lambda *z: np.column_stack(z)
basic=F(one,h,D)
models={
'hold, drift': (basic,)*3,
'hold and drift, no offset': (F(h,D),)*3,
'hold drift product': (F(basic,h*D),)*3,
'hold drift quadratic': (F(basic,h*D,D**2,h**2),)*3,
'hold drift rational': (F(basic,h*D,D/(1+D**2),h/(1+D**2)),)*3,
'hold drift gap': (F(basic,hist),)*3,
'hold drift gap products': (F(basic,hist,h*D,D[:,None]*hist),)*3,
'hold raw features': (F(one,h,x),)*3,
'hold raw features interactions': (F(one,h,x,h[:,None]*x),)*3,
'using history': (F(one,gap(tau),D),)*3,
'asymmetric simple': (F(h,D),F(one,h,D,h*D),F(one,h,D,h*D)),
'full history': (F(basic,p),)*3,
}
print('batch items',len(pickle.load(open('/holdout_task/public_batch.pkl','rb')))); print('test hold, D ranges',np.ptp(m._coupon(te,x0[n:],bridge)[1]),np.ptp(m._coupon(te,x0[n:],bridge)[2]@th[2:]))
for name,A in models.items():
 print(name); r=fit(A,print_co=True); rr=r.copy()
 for g in groups: rr[g]=fit(A,np.setdiff1d(np.arange(N),g))[g]
 print('fit',fmt(r),'LOLab',fmt(rr),'worst',np.max([stats(rr[g]) for g in groups],axis=0).round(5))
 if name=='hold, drift':
  print('mean residual over time',r.mean(0).round(5))
  print('residual correlations (first three time + final + rate)')
  covars=F(h,D,x,p,hist,h*D,D**2,h**2)
  print(np.corrcoef(covars.T,r[:,[0,1,2,-2,-1]].T)[:-5,-5:].round(2))

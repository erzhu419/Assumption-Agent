exec(open('/tmp/stress05.py').read().split('for name,i,j in slices:')[0])
from scipy.optimize import least_squares
from itertools import combinations
x=np.r_[tr['x'],s['test']['x']]; z=m._coordinates(s['panel_a'],x); y=tr['y']; F=m._fit
print('ranges', z[:72].min(0),z[:72].max(0),z[72:].min(0),z[72:].max(0))
for compact in (False,True):
 b=m._operator(s['panel_b'],z,compact=compact)
 for nm in ('1+1','1-1','52/20'):
  _,i,j=next(q for q in slices if q[0]==nm); bb=b[np.r_[i,j]]; print('old',compact,nm,[(np.mean(abs(m._scalar(bb,y[i],k)-y[j])),np.max(abs(m._scalar(bb,y[i],k)-y[j]))) for k in (False,True)])
def pred(b,i,j,key,lam):
 W=m._affine(b/b[i].std(0)); w=W[:,1:]; P=w[:,[0,1]]*w[:,[2,3]]; N=np.c_[W,P]; k=F(W[i],y[i]);
 def design(ix,p):
  h=W[ix]@p; inv=1/(1+h*h)
  if key=='full': return np.c_[N[ix],N[ix]*(inv-1)[:,None]]
  if key=='compact': return np.c_[W[ix],P[ix]*inv[:,None],W[ix]*(inv-1)[:,None]]
  if key=='pair': return np.c_[N[ix],W[ix]*(inv-1)[:,None]]
  if key=='pairden': return np.c_[W[ix],P[ix]*inv[:,None]]
  if key=='tanh': return np.c_[N[ix],np.tanh(h)-h]
  if key=='sat': return np.c_[N[ix],h*(inv-1)]
  if key=='cross': return np.c_[W[ix],P[ix],P[ix]*w[ix][:,[1,2]],P[ix]*w[ix][:,[3,0]]]
 def reg(f):
  r=np.full(f.shape[1],lam);r[:5]=1e-8;r[5:7]*=.1
  return np.diag(np.sqrt(r))
 def coeff(f):
  return F(np.r_[f,reg(f)],np.r_[y[i],np.zeros(f.shape[1])])
 def err(p):
  f=design(i,p); c=coeff(f);return np.r_[f@c-y[i],reg(f)@c,.005*p]
 starts=[k*.3,k,k*2,*(np.eye(5)[1:]*.7)]
 p=min((least_squares(err,h,bounds=(-6,6),max_nfev=90) for h in starts),key=lambda q:q.cost).x
 e=np.mean(abs(design(j,p)@coeff(design(i,p))-y[j]));return e
for compact in (False,True):
 b=m._operator(s['panel_b'],z,compact=compact); allr=[]
 for key in ('full','compact','pair','pairden','tanh','sat','cross'):
  for lam in (.003,.03,.3):
   er=np.array([pred(b,i,j,key,lam) for _,i,j in slices]); allr.append((er.max(),key,lam,er[0],er[:14].max(),[slices[t][0] for t in np.argsort(er)[-3:]],er[14:].round(4).tolist()))
 print('ranked',compact,*sorted(allr)[:12],sep='\n',flush=True)

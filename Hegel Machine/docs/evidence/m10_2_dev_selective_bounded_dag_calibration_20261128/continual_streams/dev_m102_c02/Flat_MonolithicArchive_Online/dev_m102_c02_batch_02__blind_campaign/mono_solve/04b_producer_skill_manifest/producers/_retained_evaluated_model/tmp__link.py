exec(open('/tmp/explore.py').read().split('ops={}')[0])
from scipy.optimize import least_squares
p=s['panel_b']; w=aff(z)@lstsq(aff(p['u']),p['v'])[0]; W=aff(w); P=w[:,:2]*w[:,2:]; I=np.arange(52); J=np.arange(52,len(y)); rng=np.random.default_rng(900)
splits=[(I,J)]+[(k[:52],k[52:]) for k in [rng.permutation(len(y)) for _ in range(12)]]
for nt,fn in [('tanh',np.tanh),('rat',lambda h:h/(1+h*h)),('sqrt',lambda h:h/np.sqrt(1+h*h)),('atan',np.arctan),('sin',np.sin)]:
 for m in ['tied','separate','tied+pair','fullq']:
  P1=P.sum(axis=1,keepdims=True) if m.startswith('tied') else P
  if m=='fullq':P1=poly(w,2)[:,5:]
  A=np.c_[W,P1]
  def design(q):
   h=A@q; F=np.c_[np.ones(len(h)),h,fn(h)];return np.c_[F,P] if m=='tied+pair' else F
  def fit(i,j):
   def fun(q):
    D=design(q)[i];return D@lstsq(D,y[i])[0]-y[i]
   b=lstsq(A[i],y[i])[0]; b0=[]
   for h in [.35,.7,1.25]:
    r=least_squares(fun,b*h,max_nfev=100);b0.append(r)
   r=min(b0,key=lambda r:r.cost);D=design(r.x); b=lstsq(D[i],y[i])[0];e=D@b-y;return np.mean(abs(e[i])),np.mean(abs(e[j])),r.x,b
  a=fit(I,J); reps=[fit(i,j)[1] for i,j in splits[1:]]
  print(nt,m,np.round(a[:2],6),round(np.mean(reps),6),round(max(reps),6),np.round(a[2],5),np.round(a[3],5),flush=True)
print('times',s['eval_times']);print('summary',s['train']['summary'])

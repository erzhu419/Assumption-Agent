exec(open('/tmp/readout.py').read().split('for name,f in dict')[0])
from scipy.optimize import minimize_scalar
from scipy.linalg import solve
sl,off=a._calibration(s['panel_a']); B=s['panel_b']; C=s['panel_c']; bw=np.linalg.lstsq(np.c_[np.ones(len(B['u'])),B['u']],B['v'],rcond=None)[0]; W=bw[1:]; bias=bw[0];
coords={'x':x,'inv':(x-off)/sl,'fwd':x*sl+off,'sy':u,'st':v}
for nm in ['x','inv','fwd']:
 for j,M in enumerate([W,np.linalg.inv(W),W.T,np.linalg.inv(W.T)]):coords[nm+'B'+str(j)]=coords[nm]@M+bias
folds=[np.arange(52,72)]+list(np.array_split(np.arange(n),4))
def cv(F,l=0):
 H=F@F.T; es=[]
 for j in folds:
  i=np.setdiff1d(np.arange(n),j)
  pred=F[j]@np.linalg.lstsq(F[i],y[i],rcond=1e-8)[0] if l==0 else H[np.ix_(j,i)]@solve(H[np.ix_(i,i)]+l*np.eye(len(i)),y[i],assume_a='pos')
  es.append(np.mean(abs(y[j]-pred)))
 return np.array(es)
A=C['u']; T=C['v1']; forms={'tanh':lambda U,k:np.tanh(k*U),'tanh+lin':lambda U,k:np.c_[np.tanh(k*U),U],'tanh_eps':lambda U,k:np.tanh(k*U)+.22*U,'lin_eps_tanh':lambda U,k:U+.22*np.tanh(k*U),'atan':lambda U,k:np.arctan(k*U),'sin':lambda U,k:np.sin(k*U),'rational':lambda U,k:U/(1+k*U**2),'squash':lambda U,k:U/np.sqrt(1+k*U**2)}
res=[]; Q=poly(x,2)
for name,f in forms.items():
 def loss(k):
  F=np.c_[np.ones(len(A)),f(A,k)]; return np.mean((T-F@np.linalg.lstsq(F,T,rcond=None)[0])**2)
 k=minimize_scalar(loss,bounds=(.03,3),method='bounded').x; print('panel',name,k,loss(k),loss(1.25),loss(.22)); P=np.c_[np.ones(len(A)),f(A,k)]; Wc=np.linalg.lstsq(P,T,rcond=None)[0]
 for nm,d in coords.items():
  for kk in sorted(set([k,.22,.5,1.,1.25,1.8,2.5])):
   G=f(d,kk); Gc=np.c_[np.ones(n),G]@Wc
   for bn,g in [('nl',poly(G,2)),('panel',poly(Gc,2)),('q+nl',np.c_[Q,G]),('q+diag',np.c_[Q,G,G**2]),('q+cross',np.c_[Q,G,(G[:,:,None]*x[:,None,:]).reshape(n,-1)]),('q+panel',np.c_[Q,poly(Gc,2)]),('q+mode',np.c_[Q,poly(Gc,2),z[:,0]])]:
    F=np.c_[np.ones(n),g]; F[:,1:]/=np.maximum(F[:,1:].std(0),1e-8)
    for l in [0,.001,.03,.3]:
     E=cv(F,l); res.append((max(E),E.mean(),E.round(5),name,nm,round(kk,5),bn,l))
for row in sorted(res,key=lambda r:r[0])[:28]:print(row)

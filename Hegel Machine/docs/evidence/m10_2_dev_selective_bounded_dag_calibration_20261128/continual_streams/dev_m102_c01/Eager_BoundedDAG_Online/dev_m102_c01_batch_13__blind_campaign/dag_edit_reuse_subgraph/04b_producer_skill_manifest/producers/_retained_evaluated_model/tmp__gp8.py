exec(open('/tmp/readout.py').read().split('for name,f in dict')[0])
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
D=cdist(x,x,'sqeuclidean'); E=cdist(u,u,'sqeuclidean'); G=cdist(v,v,'sqeuclidean'); T=cdist(z[:,:1],z[:,:1],'sqeuclidean'); Q=1+x@x.T; A=(x@x.T)/4; B=(u@u.T)/4
base=[np.ones_like(D),A,A*A,B,B*B,p['K']/10]
def ker(w):
 w=np.exp(w); return sum(c*k for c,k in zip(w[:6],base))+w[6]*np.exp(-w[7]*D-w[8]*E-w[9]*G-w[10]*T)+w[11]*np.exp(-w[12]*D)+w[13]*np.eye(n)
folds=[np.arange(52,72)]+list(np.array_split(np.arange(n),4))
for j in folds:
 i=np.setdiff1d(np.arange(n),j)
 def loss(w):
  H=ker(w)[np.ix_(i,i)]; c=cho_factor(H); return (y[i]@cho_solve(c,y[i])+2*np.log(c[0].diagonal()).sum())/2+.10*np.sum((w-w0)**2)
 w0=np.log([.1,.1,.2,.03,.03,.03,.2,.4,.05,.05,.002,.1,.1,.0001]); fit=minimize(loss,w0,method='L-BFGS-B',bounds=[(-14,4)]*14); H=ker(fit.x); pr=H[np.ix_(j,i)]@cho_solve(cho_factor(H[np.ix_(i,i)]),y[i]); print('GP',np.mean(abs(y[j]-pr)),np.exp(fit.x).round(5))

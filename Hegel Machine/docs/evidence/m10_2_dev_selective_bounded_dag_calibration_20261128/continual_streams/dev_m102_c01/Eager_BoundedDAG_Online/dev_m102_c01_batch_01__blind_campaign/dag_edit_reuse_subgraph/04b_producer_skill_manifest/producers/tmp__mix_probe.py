exec(open('/tmp/scalar_probe.py').read().split('models=')[0])
from scipy.linalg import cho_factor,cho_solve
D=np.maximum(0.,K.diagonal()[:,None]+K.diagonal()[None,:]-2*K)
S=st@st.T
Y=poly(sy,2)@poly(sy,2).T
folds += [np.argsort(x[:,j])[:12] for j in range(4)] + [np.argsort(x[:,j])[-12:] for j in range(4)]
def test(g,r,ic):
 out=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(N),te); mu=y[tr].mean(); A=cho_factor(g[np.ix_(tr,tr)]+r*np.eye(len(tr))); a=cho_solve(A,y[tr]-mu)
  if ic:
   b=cho_solve(A,np.ones(len(tr))); z=a.sum()/b.sum(); a-=z*b; mu+=z
  out.append(np.mean(abs(mu+g[np.ix_(te,tr)]@a-y[te])))
 return np.array(out)
configs=[('arch',K+.3*sy@sy.T,.001,False),('stq',K+30*S,.0001,False),('kd',K+3*np.exp(-D/12),.003,False)]
for s0 in [0,3,30]:
 for l in [12,24,48]:
  for a in [.3,3]:
   for b in [0,.1]:
    for r in [.0001,.0003,.001,.003]:
     for ic in [False,True]:configs.append(((s0,l,a,b),K+s0*S+a*np.exp(-D/l)+b*Y,r,ic))
res=[]
for k,g,r,ic in configs:
 o=test(g,r,ic); m=np.array([o[0],o[1:7].mean(),max(o[1:7]),o[7],o[8:].mean(),max(o[8:])]); res.append((m,k,r,ic))
for row in res[:3]:print(row)
for row in sorted([r for r in res if r[0][0]<.038],key=lambda r:r[0][1]+.2*r[0][3]+.5*r[0][4]+.2*r[0][5])[:18]:print(row)

exec(open('/tmp/routes.py').read().split('# Rank')[0])
from scipy.linalg import solve
from scipy.spatial.distance import cdist
x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; a=(x-A[0])@np.linalg.inv(A[1:]); b=col(a)@B
sp=[(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20)),(np.r_[0:20,40:72],np.arange(20,40))]
def gram(z):return z@z.T
def scores(K,alpha):
 E=[]
 for tr,te in sp:
  e=K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+alpha*np.eye(len(tr)),Y[tr],assume_a='pos')-Y[te]; E.append([np.mean(abs(e[:,0])),np.mean(abs(e[:,1:])),np.mean(abs(e[:,-1]))])
 return np.array(E)
R=[]
for name,z in [('B',b),('Bx',col(x)@B),('raw',a),('Bt',(a@B[1:].T+B[0])),('BC',C(b)),('BD',D(b)),('CB',col(C(a))@B),('DB',col(D(a))@B)]:
 for w in [.65,1,1.4,1.8,2.2,2.8,3.5]:
  for fn,h in [('tan',np.tanh(w*z)),('sin',np.sin(w*z)),('asinh',np.arcsinh(w*z)),('rat',w*z/(1+(w*z)**2)),('atan',np.arctan(w*z))]:
   for lmix in [0,.25,1,4]:
    F=gram(h)+lmix*gram(z); G=1+F
    for desc,K in [('q',G**2),('c',G**3),('qc',G**2+.1*G**3),('qz',G**2+gram(z)),('sep',1+gram(z)+gram(h)+gram(h*h)+gram(z*h)+.2*F**2),('qzp',G**2+(1+gram(z))**2)]:
     K=K/np.mean(np.diag(K))
     for al in [1e-6,.0001,.002]:
      E=scores(K,al); R.append((E.mean(0)[0],E.mean(0)[1],name,fn,w,lmix,desc,al,E.round(5).tolist()))
for k in [lambda z:z[0],lambda z:z[1],lambda z:z[0]+z[1], lambda z:z[-1][0][0]]:
 print(*sorted(R,key=k)[:8],sep='\n');print()

from graph import *
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=160)
print('A B',A,B)
W={k:Z[k] for k in ('r','a','f','rb','ab','fb')}
for n,w in list(W.items()):
 if not n.endswith('b'):
  for g in (.22,1): W[n+'s'+str(g)+'b']=af(soft(w,g))@B
R=[]
for n,w in W.items():
 for g in [0,.04,.09,.15,.22,.35,.5,.75,1,1.5,2]:
  z=soft(w,g); G=basis(z,3)
  D={'sum2':(z*z).sum(1),'wsum2':(w*w).sum(1),'sum':z.sum(1)**2,'prod':z.prod(1)**2}
  for i in range(4):
   D[str(i)]=z[:,i]**2
   for j in range(i):
    for s in [-1,1]:D[f'{i}{s:+}{j}']=(z[:,i]+s*z[:,j])**2
  for dn,d in D.items():
   for a in [.1,.22,.5,1.25]:
    F=np.c_[np.ones(len(G)),G/(1+a*d[:,None])]; r=score(F,Y,.003); R.append((r,(n,g,dn,a)))
for it in [0,1,2,3,4,5]:
 print('RANK',it,*[(l,np.round(r[:6],4)) for r,l in sorted(R,key=lambda t:t[0][it])[:7]],sep='\n')
for n in ['r','a','f','rb','ab','fb','as0.22b','fs0.22b']:
 w=W[n]; scores=[]
 for d in [2,3]:
  for target in [y,T@np.linalg.svd(T[:52],full_matrices=False)[2][0]]:
   E=[]; ps=[]
   for i,j in splits[:4]:
    def matrix(p):
     z=soft(w,np.exp(p[0])); F=basis(z,d); q=(z@p[1:5]+p[5])**2
     return np.c_[np.ones(len(z)),F/(1+q[:,None])]
    def res(p,out=False):
     F=matrix(p); P=F[i]; pen=.002; reg=np.r_[0,np.repeat(pen,F.shape[1]-1)]; C=np.linalg.solve(P.T@P+np.diag(reg),P.T@target[i]); V=F@C
     return V if out else np.r_[V[i]-target[i],np.sqrt(reg)*C,.009*p[1:]]
    init=[np.r_[-2.,k,0.] for k in [np.ones(4)*.2,*list(np.eye(4)*.5)]]
    opt=min([least_squares(res,p,bounds=([-8]+[-4]*5,[3]+[4]*5),max_nfev=90) for p in init],key=lambda p:p.cost); v=res(opt.x,True); E.append(np.mean(abs(v[j]-target[j])));ps.append(opt.x)
   scores.append((d,np.round(E,4),np.round(ps[0],3),np.round(np.std(ps,0),3)))
 print('profile',n,*scores,flush=True)

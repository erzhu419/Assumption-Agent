from operators import *
from itertools import combinations_with_replacement,combinations
from scipy.optimize import minimize_scalar
c=s['panel_c']; u=c['u']; pairs=list(combinations(range(4),2)); cross=lambda t:np.c_[[t[:,i]*t[:,j] for i,j in pairs]].T
C1=fit(F(cross(u)),c['v1']); C1[abs(C1)<.0005]=0
cos=lambda t,g:F(np.cos(g*t)); cost=lambda g:np.linalg.norm(cos(u,g)@fit(cos(u,g),c['v2'])-c['v2']); g=minimize_scalar(cost,bounds=(1.5,3.3),method='bounded').x; C2=fit(cos(u,g),c['v2']); print('operators',C1,C2,g)
def f(t):return F(cross(t))@C1
def h(t):return cos(t,g)@C2
x=s['train']['x']; z=(x-a)/b; Y=np.c_[s['train']['y'],s['train']['traj']]; n=40
Yred=np.c_[Y[:,0],Y[:,1:]@np.linalg.svd(Y[:n,1:],full_matrices=False)[2][:3].T]; Yred/=Yred[:n].std(axis=0)
ones=np.ones((len(x),1)); results=[]
def test(M,key):
 if M.shape[1]>42:return
 Sd=np.maximum(M[:n].std(0),.001); M=M/Sd; U,sv,V=np.linalg.svd(M[:n],full_matrices=False); k=sv>sv[0]*.00015
 if k.sum()>36:return
 R=abs(M@((V[k].T/sv[k])@(U[:,k].T@Y[:n]))-Y); r0=np.mean(R[:n],0);r1=np.mean(R[n:52],0);r2=np.mean(R[52:],0); results.append((r1[0],r1[1:].mean(),r2[0],r2[1:].mean(),r0[0],r0[1:].mean(),key))
def out(A,D):return (A[:,:,None]*D[:,None,:]).reshape(len(x),-1)
def powers(t,k):return np.stack([np.prod(t[:,inds],axis=1) for inds in combinations_with_replacement(range(t.shape[1]),k)],axis=1)
for name,w in {'z':z,'Bz':Bf(z),'Binvz':Binv(z),'x':x,'Blinz':z@B[1:]}.items():
 ca,cb=f(w),h(w); L={'w':w,'z':z,'ca':ca,'cb':cb,'hca':h(ca),'fcb':f(cb),'fa':f(ca),'hb':h(cb),'Bca':Bf(ca),'Bcb':Bf(cb),'fBca':f(Bf(ca)),'hBca':h(Bf(ca)),'fBcb':f(Bf(cb)),'hBcb':h(Bf(cb))}; P=np.c_[ones,w,powers(w,2)]; P3=np.c_[P,powers(w,3)]; tests=[('P0',ones),('P1',F(w)),('P2',P),('P3',P3)]
 for k,v in L.items():
  for k2,v2 in tests:
   test(np.c_[v2,v],(name,k2,k));test(np.c_[v2,v,powers(v,2)],(name,k2,k,'quad'))
  for l,d in L.items():
   base=np.c_[ones,w,ca,cb,v,d]
   for m,p in [('dot',np.sum(v*d,axis=1)[:,None]),('outer',out(v,d)),('diag',v*d)]:test(np.c_[base,p],(name,k,l,m))
   for gain in [.22,1.25]:
    S=np.sum(d*d,axis=1,keepdims=True)
    for m,p in [('rat',v/(1+gain*d*d)),('nrat',v/(1+gain*S)),('exp',v*np.exp(-gain*d*d)),('tanh',v*np.tanh(gain*d)),('cos',v*np.cos(gain*d)),('sig',v/(1+np.exp(-gain*d))),('mix',(v+d)/(1+gain*(v-d)**2)),('sinc',v*np.sinc(gain*d/np.pi))]:
     test(np.c_[base,p],(name,k,l,m,gain));test(np.c_[P3,p],(name,'p3',k,l,m,gain))
 print('done',name)
for k in [0,1,4,5]:print('best',k,*sorted(results,key=lambda j:j[k])[:10],sep='\n')

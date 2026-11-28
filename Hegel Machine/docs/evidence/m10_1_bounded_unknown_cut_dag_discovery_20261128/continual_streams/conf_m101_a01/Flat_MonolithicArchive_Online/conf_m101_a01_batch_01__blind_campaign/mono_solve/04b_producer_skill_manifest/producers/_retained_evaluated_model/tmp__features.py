import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=t['y']; tr=t['traj']; tm=s['eval_times']
def af(x):return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def pol(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for j in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),j)])
p=s['panel_a']; a=af(p['u']); z=p['v']; good=np.linalg.norm(z-a@ls(a,z),axis=1)<3; A=ls(a[good],z[good]); p=s['panel_b']; B=ls(af(p['u']),p['v']); p=s['panel_c']; C=ls(af(p['u']),p['v1']); D=ls(pol(p['u'],2),p['v2'])
V=np.linalg.svd(tr[:52],full_matrices=False)[2]; v=V[0]/V[0,-1]; h=tr@v/(v@v); Y=np.c_[y,h]; n=len(y)
coords={'x':x,'a':af(x)@A,'ai':(x-A[0])@np.linalg.inv(A[1:])}
for k,u in list(coords.items()):
 for k2,M in [('b',B),('c',C)]:coords[k+k2]=af(u)@M
for k,u in list(coords.items()):
 if len(k)<=2:coords[k+'2']=pol(u,2)@D
splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [np.random.default_rng(i).permutation(n) for i in range(10)]]
def cv(F,reg=0,base=None):
 if base is None:base=af(x)
 r=[]
 for i,j in splits:
  m=ls(base[i],F[i]); G=F-base@m; scale=np.sqrt(np.mean(G[i]**2,axis=0)); G=G/np.maximum(scale,1e-10); H=np.c_[base,G]; P=np.r_[np.zeros(base.shape[1]),np.full(F.shape[1],reg)]; w=ls(np.r_[H[i],np.diag(np.sqrt(P))],np.r_[Y[i],np.zeros((len(P),2))]); r.append(np.mean(abs(H[j]@w-Y[j]),0))
 r=np.array(r);return r[0],np.median(r[1:],0),np.max(r[1:],0)
if __name__=='__main__':
 np.set_printoptions(precision=5,suppress=True); print('B',B,'C',C,'D',D,sep='\n'); print('x ranges',x.min(0),x.max(0),'test',s['test']['x'].min(0),s['test']['x'].max(0)); print('Y quantile',np.quantile(Y,[0,.05,.25,.5,.75,.95,1],axis=0)); print('y cubic coeff',ls(pol(x,3),y))
 results=[]
 base=pol(x,2)
 for name,u in coords.items():
  for f,F in [('cube',u**3),('powers',np.c_[u**3,u**4,u**5]),('all3',pol(u,3)[:,15:]),('s2',np.sin(2*u)),('s3',np.sin(3*u)),('tanh',np.tanh(u)),('tanh2',np.tanh(2*u)),('exp',np.exp(u)),('rat',u/(1+u*u)),('har',np.c_[np.sin(1.6*u),np.cos(1.6*u)]),('poly2',pol(u,2)[:,5:]),('har3',np.c_[np.sin(3*u),np.cos(3*u)])]:
   for reg in [.01,.3,3]:
    m=cv(F,reg,base); results.append((m,name,f,reg))
 for j in range(2):
  print('best column',j)
  for m,*key in sorted(results,key=lambda a:a[0][0][j]+a[0][1][j])[:14]:print(key,np.array(m))
 print('relaxation')
 for typ in range(4):
  def basis(p):
   r=np.exp(p[0]); f= (1-np.exp(-r*tm)) if typ<2 else tm/(tm+r)
   if typ%2:f=f+p[1]*tm
   return f/f[-1]
  q=least_squares(lambda p:basis(p)-v,[0,0][:1+typ%2]);print(typ,q.x,q.fun)

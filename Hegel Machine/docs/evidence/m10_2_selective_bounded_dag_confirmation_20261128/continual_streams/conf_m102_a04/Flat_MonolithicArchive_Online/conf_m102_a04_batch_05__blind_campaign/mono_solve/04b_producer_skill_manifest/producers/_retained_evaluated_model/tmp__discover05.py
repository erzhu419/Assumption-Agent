import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x=d['x']; t=s['eval_times']; T=d['traj']; y=d['y']; n=52
np.set_printoptions(precision=5,suppress=True,linewidth=150)
def lin(x,y):return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def one(x):return np.c_[np.ones(len(x)),x]
def basis(r):return -np.expm1(-t[:,None]*r)
r=least_squares(lambda r:(basis(r)@lin(basis(r),T[:n].T)-T[:n].T).ravel(),[.4,1.8],bounds=([.01,.7],[.8,7]),gtol=1e-13).x
F=basis(r); amp=lin(F,T.T).T; Y=np.c_[y,amp]; print('rates',r)
p=s['panel_a']; u,v=p['u'],p['v']; keep=np.ones(len(u),bool)
for _ in range(5):
 f=one(u); e=np.linalg.norm(f@lin(f[keep],v[keep])-v,axis=1); keep=e<4*np.median(e)
def cal(q,j):return one(np.c_[q,q[:,j]*q[:,(j+1)%4],q[:,j]**3])
W=np.stack([lin(cal(u,j)[keep],v[keep,j]) for j in range(4)])
def forward(q):return np.column_stack([cal(q,j)@W[j] for j in range(4)])
print('A',W, 'residual',np.max(abs(forward(u)[keep]-v[keep])), 'kept',sum(keep));z=np.stack([least_squares(lambda a:forward(a[None,:])[0]-row,lin(W[:,1:5].T,np.eye(4)).T@(row-W[:,0])).x for row in x]);print('inverse',np.max(abs(forward(z)-x)), 'range',z.min(0),z.max(0))
p=s['panel_b'];B=lin(one(p['u']),p['v']);w=one(z)@B;wi=(z-B[0])@np.linalg.inv(B[1:]); p=s['panel_c']; u=p['u'];
def g(q):return one(q[:,[0,0,1,2]]*q[:,[1,3,2,3]])
def h(q):return one(np.cos(2.4*q))
G=lin(g(u),p['v1']);H=lin(h(u),p['v2']); print('G',G,'H',H)
def report(a,bb,tag):
 P=(a[:,:,None]*bb[:,None,:]).reshape(len(a),-1); S=np.c_[a,bb]; D=a*bb
 results=[]
 for name,f in [('base',S),('dot',np.c_[S,D.sum(1)]),('diag',np.c_[S,D]),('outer',np.c_[S,P]),('pureOuter',P),('pureDot',D.sum(1)),('dot+tanh',np.c_[S,D.sum(1),np.tanh(D.sum(1))]),('outer+square',np.c_[S,P,a*a,bb*bb]),('all',np.c_[S,P,a*a,bb*bb,z,w])]:
  f=one(f); coef=lin(f[:n],Y[:n]); pred=f@coef; ty=abs(pred[:,0]-y); tt=abs(pred[:,1:]@F.T-T); results.append((name,*np.round([ty[n:].mean(),tt[n:].mean(),ty[:n].mean(),tt[:n].mean()],5)))
 print(tag,results)
for nm,a in [('z',z),('w',w),('wi',wi),('raw',x)]:
 for nn,b in [('z',z),('w',w),('wi',wi),('raw',x)]:
  report(g(a)@G,h(b)@H,nm+' '+nn)

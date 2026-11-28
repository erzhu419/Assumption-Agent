import numpy as np,pickle,itertools,glob,runpy
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; N=len(tr['x']); I=np.ones(N); F=lambda x:np.c_[np.ones(len(x)),x]; L=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]; cv=np.arange(52); ve=np.arange(52,72)
def pairs(u):return np.column_stack([u[:,i]*u[:,j] for i,j in itertools.combinations(range(4),2)])
def cal(p,x):
 u,v=p['u'],p['v']; e=np.linalg.norm(v-F(u)@L(F(u),v),axis=1); k=e<3*np.median(e)
 def design(z,r,j):return np.c_[np.ones(len(z)),z[:,j],np.tanh(r[0]*z[:,j]),np.sin(r[1]*z[:,(j+2)%4])]
 def fit(r):return [L(design(u[k],r,j),v[k,j]) for j in range(4)]
 def forward(z,r,b):return np.column_stack([design(z,r,j)@b[j] for j in range(4)])
 r=least_squares(lambda r:(forward(u[k],r,fit(r))-v[k]).ravel(),[1.9,2.3]).x; b=fit(r); print('cal',r,b,np.std(forward(u[k],r,b)-v[k]));return np.array([least_squares(lambda z:forward(z[None],r,b)[0]-row,row).x for row in x])
def operator(fun,u,v,bounds=(.2,3)):
 def obj(r):
  a=fun(u,r);q=np.linalg.qr(a)[0];d=v-q@(q.T@v);return np.mean((d/(1-(q*q).sum(1))[:,None])**2)
 rr=minimize_scalar(obj,bounds=bounds,method='bounded'); a=fun(u,rr.x); b=L(a,v); print('op',rr.x,np.sqrt(rr.fun),np.std(a@b-v)); return lambda z:fun(z,rr.x)@b
z=cal(s['panel_a'],tr['x']); p=s['panel_b']; m=L(F(p['u']),p['v']); B=lambda x:F(x)@m
p=s['panel_c'];u=p['u'];v1=p['v1'];v2=p['v2']; soft=lambda x:x/np.sqrt(1+x*x)
A=operator(lambda x,r:np.c_[F(np.tanh(r*x)),x],u,v1); AA=operator(lambda x,r:np.c_[F(np.tanh(r*x)),x,x**3],u,v1); C=operator(lambda x,r:np.c_[F(soft(r*x)),pairs(x)],u,v2)
q=np.linalg.svd(tr['traj'][:52],full_matrices=False)[2][:2]; Y=np.c_[tr['y'],tr['traj']@q.T]; outs=[]
def score(f):
 b=L(f[cv],Y[cv]); r=f[ve]@b-Y[ve];return np.mean(abs(r[:,0])),np.mean(abs(r[:,1:]@q)),np.std(f[cv]@b-Y[cv],axis=0)
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
for xn,x in [('raw',tr['x']),('cal',z),('bcal',B(z)),('binvcal',(z-m[0])@np.linalg.inv(m[1:]))]:
 for an,a in [('A',A),('AA',AA)]:
  for bn,b in [('C',C),('BC',lambda x:B(C(x))),('B',B),('x',lambda x:x)]:
   a1,b1=a(x),b(x); diag=a1*b1
   for fn,f in [('linear',F(np.c_[a1,b1])),('sum',F(np.c_[a1,b1,diag.sum(1)])),('diag',F(np.c_[a1,b1,diag])),('full',F(np.c_[a1,b1,cross(a1,b1)])),('quadratic',F(np.c_[a1,b1,a1*a1,b1*b1,pairs(a1),pairs(b1),diag])),('latent',F(np.c_[x,a1,b1,diag])),('latentfull',F(np.c_[x,a1,b1,cross(a1,b1)]))]:
    er=score(f);outs.append((*er,xn,an,bn,fn))
for i in (0,1):
 print('rank',i)
 for o in sorted(outs,key=lambda o:o[i])[:15]: print(o)

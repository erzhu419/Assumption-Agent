import numpy as np,pickle,itertools
from scipy.optimize import least_squares, minimize_scalar
from scipy.linalg import qr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=52;x=tr['x'];t=s['eval_times'];zout=tr['traj'];
def tb(r):return -np.expm1(-t[:,None]*r)
def tres(r):
 q=tb(r);return (q@np.linalg.lstsq(q,zout[:n].T,rcond=None)[0]-zout[:n].T).ravel()
q=tb(least_squares(tres,[.4,1.85],bounds=([.1,.9],[.9,4])).x); Y=np.column_stack([tr['y'],np.linalg.lstsq(q,zout.T,rcond=None)[0].T]);
def ls(F,Y):return np.linalg.lstsq(F,Y,rcond=1e-10)[0]
def aff(u):return np.column_stack([np.ones(len(u)),u])
def poly(z,d=2):return np.column_stack([np.prod(z[:,j],axis=1) for k in range(d+1) for j in itertools.combinations_with_replacement(range(4),k)])
p=s['panel_a'];u=p['u'];v=p['v'];F=aff(u);M=ls(F,v);keep=np.linalg.norm(F@M-v,axis=1)<3;M=ls(F[keep],v[keep]);z=(x-M[0])@np.linalg.inv(M[1:]);p=s['panel_b'];B=ls(aff(p['u']),p['v']);p=s['panel_c'];u=p['u'];v1=p['v1'];v2=p['v2'];W1=ls(poly(u),v1)
print('panel1 quadratic coefficients\n',W1.round(3));
for label,phi in [('exp',lambda z,a:np.expm1(-a*z*z)),('cos',lambda z,a:np.cos(a*z)-1),('rat',lambda z,a:z*z/(1+a*z*z)),('tanh',lambda z,a:np.tanh(a*z*z)),('sqrt',lambda z,a:np.sqrt(1+a*z*z)-1)]:
 def f(z,a):return np.column_stack([aff(z),phi(z,a)])
 def loss(a):
  F=f(u,a); return np.mean((F@ls(F,v2)-v2)**2)
 r=minimize_scalar(loss,bounds=(.005,5),method='bounded');print(label,r.x,np.sqrt(r.fun));
 if label=='exp': a=r.x; W2=ls(f(u,a),v2);print('panel2 exp coefficients\n',W2.round(3))
def g2(z):return np.column_stack([aff(z),np.expm1(-a*z*z)])@W2
zs={'raw':x,'A':z,'B':aff(z)@B,'Bi':(z-B[0])@np.linalg.inv(B[1:])};
for k,w in list(zs.items()):zs[k+'1']=poly(w)@W1;zs[k+'2']=g2(w)
for k,w in list(zs.items()):
 if not k.endswith(('1','2')):
  zs[k+'mix1']=w+poly(w)@W1;zs[k+'mix2']=w+g2(w);zs[k+'sat']=w+np.expm1(-a*w*w)
def report(F):
 coef=ls(F[:n],Y[:n]);P=F[n:]@coef;return np.array([abs(P[:,0]-Y[n:,0]).mean(),abs((P[:,1:]-Y[n:,1:])@q.T).mean(),abs(F[:n]@coef-Y[:n]).mean()])
R=[]
for k,w in zs.items():
 for d in (1,2):R.append((k+str(d),report(poly(w,d))))
 if not k.endswith(('1','2')):
  j=np.column_stack([poly(w),poly(w)@W1,g2(w)]);R.append((k+'joint',report(j))); R.append((k+'w+V2 cross',report(np.column_stack([j,np.einsum('ni,nj->nij',w,g2(w)).reshape(len(w),-1)]))))
  for typ,F in [('tanh',np.tanh(w)),('sin',np.sin(w)),('rbf',np.exp(-w*w)),('rat',w/(1+w*w))]:R.append((k+typ,report(np.column_stack([poly(w),F,poly(F)]))))
for k,v in sorted(R,key=lambda item:max(item[1][:2]))[:26]:print(k,v.round(5))
print('panel2 residual by column',np.sqrt(np.mean((g2(u)-v2)**2,0)))
print('gain and offset',np.diag(M[1:]),M[0]); print('t rates',least_squares(tres,[.4,1.85]).x)

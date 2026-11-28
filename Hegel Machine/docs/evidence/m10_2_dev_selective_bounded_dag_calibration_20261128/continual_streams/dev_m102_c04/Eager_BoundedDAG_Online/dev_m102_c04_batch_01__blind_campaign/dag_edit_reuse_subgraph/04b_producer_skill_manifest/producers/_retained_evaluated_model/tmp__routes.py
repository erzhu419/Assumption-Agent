exec(open('/tmp/spatial.py').read().split("print('v1 coefs'")[0])
from itertools import product
from scipy.optimize import minimize_scalar
bp=s['panel_b']; bw=ls(affine(bp['u']),bp['v']); b0,bm=bw[0],bw[1:];
def forward(z):return z@bm+b0
def inverse(z):return np.linalg.solve(bm.T,(z-b0).T).T
v1=p['v1']; sel=np.linalg.norm(ls(np.c_[affine(u),terms(u,pairs)],v1),axis=1)[5:]>.01
f1=lambda z:np.c_[np.ones(len(z)),terms(z,pairs)[:,sel]]
w1=ls(f1(u),v1); p1=lambda z:f1(z)@w1
def f2(z,k,typ):
 q=z*z; h=q/(1+k*q)
 if typ==0:return np.c_[np.ones(len(q)),2*h-q]
 if typ==1:return np.c_[np.ones(len(q)),q,h]
 if typ==2:return np.c_[np.ones(len(q)),1-(q-1)**2/(1+k*q)]
 if typ==3:return np.c_[np.ones(len(q)),(q-1)/(1+k*(q-1)**2)]
 if typ==4:return np.c_[np.ones(len(q)),(q-1)**2/(1+k*(q-1)**2)]
 if typ==5:return np.c_[np.ones(len(q)),(q-1)**2/(1+k*q)]
 if typ==6:return np.c_[np.ones(len(q)),(q-1)/(1+k*q*q)]
 if typ==7:return np.c_[np.ones(len(q)),(1-q)/(1+k*q)**2]
for typ in range(8):
 def err(k):
  f=f2(u,k,typ); return np.linalg.norm(f@ls(f,v)-v)
 opt=minimize_scalar(err,bounds=(.005,4),method='bounded'); k=opt.x; w=ls(f2(u,k,typ),v); print('form',typ,k,opt.fun,flush=True)
 if typ==1: r1=lambda z,k=k,w=w:f2(z,k,1)@w
print('panel x ranges',[(i,np.quantile(np.linalg.norm(z,axis=1),[0,.5,1])) for i,z in [('u',u),('x',x),('B+x',forward(x)),('B-x',inverse(x))]])
Y=np.c_[t['y'],t['traj']]; n=52
raw=t['x']; allz={'a':x,'x':raw,'aB+':forward(x),'aB-':inverse(x),'xB+':forward(raw),'xB-':inverse(raw),'aB0':x@bm,'aB1':np.linalg.solve(bm.T,x.T).T}; ans=[]
for i,j,ap,bp in product(allz,allz,range(3),range(3)):
 a=p1(allz[i]); b=r1(allz[j]); a=(a,forward(a),inverse(a))[ap]; b=(b,forward(b),inverse(b))[bp]
 c,off=old['_features'](a,b); diag=a*b
 for m,f in enumerate([c,np.c_[c,off],np.c_[a,b,diag],np.c_[a,b,diag,off],np.c_[c,x],np.c_[c,x,off],np.c_[a,b,diag,x]]):
  A=affine(f); e=abs(A[n:]@ls(A[:n],Y[:n])-Y[n:]); ans.append((e[:,0].mean(),e[:,1:].mean(),i,j,ap,bp,m))
for key in (lambda a:a[0]+a[1],lambda a:a[0],lambda a:a[1]):print(sorted(ans,key=key)[:8])

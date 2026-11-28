import numpy as np
from fit14 import aff,fit
from scipy.optimize import least_squares
M=np.load('/tmp/m14.npz'); z,b,c,d=[M[k] for k in 'zbcd']; y=M['Y'][:,0]; n=52
h=aff(np.c_[b,(b*np.roll(z,-1,axis=1)).sum(1)]); h4=aff(np.c_[b,b*np.roll(z,-1,axis=1)])
q=np.sign(z)*np.maximum(abs(z)-.35,0)**1.5
pb=h@fit(h[:n],y[:n]); vv=b*np.roll(z,-1,axis=1); p=pb-vv.sum(1)*fit(h[:n],y[:n])[-1];
def score(h):
 u,s,v=np.linalg.svd(h[:n],full_matrices=False); good=s>s[0]*1e-10; u,s,v=u[:,good],s[good],v[good]; pred=h@(v.T@((u.T@y[:n])/s)); e=y-pred; return np.mean(abs(e[:n]/(1-(u*u).sum(1)))),np.mean(abs(e[n:])),np.mean(abs(e[:n]))
print('base',score(h),score(h4),fit(h[:n],y[:n])); rows=[]
def add(k,x,base=h):
 a=np.c_[base,x];rows.append((*score(a),k,a))
for sk,S in [('z',z),('q',q),('b',b),('c',c),('d',d),('p',p[:,None]),('pb',pb[:,None])]:
 for i,v in enumerate(S.T):
  for l in [.1,.22,.5,1,1.25,2.15,4,8,16]:
   for fk,f in [('div',1/(1+l*v*v)),('exp',np.exp(-l*v*v)),('tan',np.tanh(l*v)),('sin',np.sin(l*v)),('soft',np.logaddexp(0,l*v)),('sq',v*v)]:
    for tk,X in [('one',np.ones((len(v),1))),('b',b),('q',q),('bz',np.c_[b,z]),('p',p[:,None]),('pb',pb[:,None]),('vv',vv.sum(1)[:,None])]:add((sk,i,l,fk,tk),X*f[:,None])
for k,X in [('z',z),('q',q),('b',b),('c',c),('d',d)]:
 for l in [.22,.5,1,1.25,2.15,3,4,6]:
  for nk,U in [('self',X),('b',b),('q',q),('z',z)]:
   for i in range(4):
    S=np.roll(X,i,axis=1)
    for fk,f in [('div',U/(1+l*S*S)),('tan',U*np.tanh(l*S)),('sq',S*S),('cub',S**3),('sin',np.sin(l*S)),('tanh',np.tanh(l*S)),('soft',np.logaddexp(0,l*S)),('cos',np.cos(l*S))]:add((k,l,nk,i,fk),f)
for name,X in [('b',b),('q',q),('z',z),('c',c),('d',d),('p',p[:,None])]:
 for name2,W in [('b',b),('q',q),('z',z),('c',c),('d',d),('p',p[:,None])]:
  for i,v in enumerate(X.T):
   for j,w in enumerate(W.T):add(('pair',name,i,name2,j),v*w)
for a,b,c,k,_ in sorted(rows,key=lambda r:r[0])[:28]:print(np.round([a,b,c],6),k)
print('best standalone threshold',[(k,score(np.c_[h,x])) for k,x in [('q1',q[:,1]),('z1',z[:,1]),('q1div',q[:,1]/(1+2.15*z[:,1]**2))]])
for x in [z[:,1],q[:,1],b[:,2],p,pb,vv.sum(1)]:
 print('nonlinear',end=' ')
 for fn in [lambda p:np.tanh(p[0]*(x-p[1])),lambda p:np.sin(p[0]*(x-p[1])),lambda p:x/(1+p[0]*(x-p[1])**2)]:
  def design(p):return np.c_[h,fn(p)]
  def r(p):
   D=design(p)[:n];return D@fit(D,y[:n])-y[:n]
  v=least_squares(r,[1,0],bounds=([.02,-3],[20,3])).x; print(v,score(design(v)),end=' ')
 print()

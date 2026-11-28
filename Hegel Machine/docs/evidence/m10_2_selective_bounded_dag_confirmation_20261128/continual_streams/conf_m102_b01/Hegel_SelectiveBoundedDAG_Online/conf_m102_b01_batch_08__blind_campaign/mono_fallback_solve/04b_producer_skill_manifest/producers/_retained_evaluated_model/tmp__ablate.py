import numpy as np,pickle,sys,itertools
sys.path.insert(0,'/holdout_task');import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; X=t['x'];Y=np.c_[t['y'],t['traj']]; N=len(X); z=m._calibrate(s['panel_a'],X); B,C,D=m._branches(s,X); p=s['panel_c']; U=p['u']; q=m._ls(m._pairmap(U),p['v1']); print('C1 error',np.std(m._pairmap(U)@q-p['v1'])); c1=lambda x:m._pairmap(x)@q; c2=lambda x:m._project(U,p['v2'],x,lambda u,q:u+q[0]*u*u+q[1]*(1-np.cos(q[2]*u)),[.35,.065,2.5]);
T=np.r_[X,s['test']['x']]; BB,CC,DD=m._branches(s,T);print('range train/test',[(np.max(abs(f[:72]),0).round(2),np.max(abs(f[72:]),0).round(2)) for f in [T,BB,CC,DD]]); print('C,D stats',C.mean(0),C.std(0),D.mean(0),D.std(0));
splits=[np.arange(k,k+20) for k in (52,0,20,40)]+[np.random.default_rng(k).permutation(N)[:20] for k in range(8)]; inds=[(np.setdiff1d(np.arange(N),te),te) for te in splits]
F=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def gram(f,tr):
 g=(f-f[tr].mean(0))/np.maximum(f[tr].std(0),.0001);return g@g.T/g.shape[1]
def test(c,d,b,w,reg,kind):
 E=[]
 for tr,te in inds:
  l=gram(np.c_[c,d],tr); p=gram(F(c,d),tr); k=10+l+p+w*gram(b,tr)
  if kind=='poly':k+=.1*l*l+.07*l*l*l
  if kind=='prod':k+=.2*l*p
  if kind=='mix':k+=.05*l*l+.03*l*l*l+.10*l*p
  if kind=='more':k+=.2*l*l+.2*l*p+.05*p*p
  if kind=='rbf':
   v=np.diag(l); dist=v[:,None]+v[None,:]-2*l;k+=.5*np.exp(-dist*.25)
  if kind=='diag':k+=1.25*gram(c*d,tr)+.15*l*p
  if kind=='diag_only':k+=1.25*gram(c*d,tr)-.8*p+.15*l*p
  mu=Y[tr].mean(0);e=abs(mu+k[np.ix_(te,tr)]@np.linalg.solve(k[np.ix_(tr,tr)]+reg*np.eye(len(tr)),Y[tr]-mu)-Y[te]); E.append([e[:,0].mean(),e[:,1:].mean(), e[:,-1].mean()])
 return np.array(E)
res=[]
for (cn,Cx),(dn,Dx) in itertools.product([('B',C),('z',c1(z)),('B0',c1(B-B.mean(0)))],[('z',D),('B',c2(B)),('C',c2(C))]):
 for (a,c),(b,d),(bn,bs),w,r,k in itertools.product([('atan',np.arctan(1.25*Cx)),('tanh',np.tanh(1.25*Cx)),('asinh',np.arcsinh(2*Cx)),('raw',Cx)],[('tanh',np.tanh(1.25*Dx)),('raw',Dx),('atan',np.arctan(1.25*Dx))],[('B',B),('z',z),('Bz',np.c_[B,z]),('z+B',np.c_[B,z,np.tanh(1.25*B),np.tanh(1.25*z)])],(.12,.4),(.003,.012,.035),('poly','prod','mix','more','rbf','diag','diag_only')):
  if (cn,dn)!=('B','z') and (bn!='B' or w!=.4 or r!=.012):continue
  e=test(c,d,bs,w,r,k); loss=e.mean(0)+.45*e.max(0);res.append((loss,(cn,dn,a,b,bn,w,r,k),e))
for j in (0,1):
 print('TARGET',j)
 for l,name,e in sorted(res,key=lambda v:v[0][j])[:14]: print(name,'gate',e[0].round(4),'mean',e.mean(0).round(4),'worst',e.max(0).round(4))

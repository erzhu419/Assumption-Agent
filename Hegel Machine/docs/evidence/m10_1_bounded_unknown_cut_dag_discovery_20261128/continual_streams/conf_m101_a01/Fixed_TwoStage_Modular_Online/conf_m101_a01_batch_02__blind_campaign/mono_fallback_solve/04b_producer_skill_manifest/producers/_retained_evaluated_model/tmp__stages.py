exec(open('/tmp/probe.py').read().split("print('A',")[0])
from scipy.optimize import least_squares
c=s['panel_c'];u=c['u']; b=s['panel_b']; B=reg(af(b['u']),b['v']); 
# Fit separable panel maps by variable projection.
def bases(z,k,p):
 if k=='soft': return z/(1+(p*z)**2)**.5
 if k=='tanh': return np.tanh(p*z)
 if k=='rat': return z/(1+(p*z)**2)
 if k=='atan': return np.arctan(p*z)
 if k=='cos': return np.cos(p*z)
 if k=='sin': return np.sin(p*z)
models={}
for key in ('v1','v2'):
 v=c[key]; out=[]
 for k in ('soft','tanh','rat','atan','cos','sin'):
  for a in (0,1):
   def F(z,p):return np.column_stack([np.ones(len(z))]+([z] if a else [])+[bases(z,k,p)])
   for width in (1,4):
    def residual(p):
     h=F(u,p);return (h@reg(h,v)-v).ravel()
    opt=min([least_squares(residual,np.full(width,p),bounds=(.25,5),max_nfev=100) for p in (1.,2.2,3.5)],key=lambda o:np.linalg.norm(o.fun)); p=opt.x; ww=reg(F(u,p),v)
    out.append((np.std(opt.fun),k,a,p,ww))
 out.sort(key=lambda z:z[0]);print(key,'fits',[(r,k,a,p) for r,k,a,p,_ in out[:8]]);r,k,a,p,ww=out[0];print('weights',ww, 'nonlinear svd',np.linalg.svd(ww[-4:])[1]);models[key]=(k,a,p,ww)
def op(z,key):
 k,a,p,ww=models[key]; return np.column_stack([np.ones(len(z))]+([z] if a else [])+[bases(z,k,p)])@ww
Aoff=w[0];Ascale=np.diag(w[1:]);Ai=lambda z:(z-Aoff)/Ascale; A=lambda z:z*Ascale+Aoff; Bf=lambda z:af(z)@B; Bi=lambda z:(z-B[0])@np.linalg.inv(B[1:]);
root={'x':x,'a':A(x),'i':Ai(x),'b':Bf(x),'j':Bi(x),'ib':Bf(Ai(x)),'ij':Bi(Ai(x)),'bi':Ai(Bf(x)),'ji':Ai(Bi(x))}
z=dict(root)
for n,v in root.items():
 for k in ('v1','v2'):
  zz=op(v,k);z[n+k[-1]]=zz;z[n+k[-1]+'b']=Bf(zz);z[n+k[-1]+'j']=Bi(zz)
train=[np.arange(52),np.arange(16,72),np.r_[0:27,43:72],np.r_[0:40,56:72]];test=[np.setdiff1d(np.arange(72),j) for j in train]
Y=np.c_[y,tr];scores=[]
def cv(F,lam):
 F=F.copy();F[:,1:]/=np.maximum(F[:,1:].std(0),.03)
 k=F@F.T; out=[]
 for i,j in zip(train,test):out.append(np.mean(abs(k[np.ix_(j,i)]@np.linalg.solve(k[np.ix_(i,i)]+lam*np.eye(len(i)),Y[i])-Y[j]),0))
 m=np.asarray(out);return m[:,0],m[:,1:].mean(1)
for n,v in z.items():
 for desc,f in [('lin',af(v)),('q',poly(v,2)),('c',poly(v,3)),('x+v',np.c_[af(x),v]),('x+qv',np.c_[poly(x,2),poly(v,2)[:,1:]]),('q(x,v)',poly(np.c_[x,v],2)),('xq+v',np.c_[poly(x,2),v])]:
  for lam in (.002,.05,.5):
   yy,ss=cv(f,lam);scores.append((yy.mean(),ss.mean(),n,desc,lam,yy,ss))
for idx in (0,1):
 print('best',idx)
 for r in sorted(scores,key=lambda a:a[idx])[:18]:print(r)

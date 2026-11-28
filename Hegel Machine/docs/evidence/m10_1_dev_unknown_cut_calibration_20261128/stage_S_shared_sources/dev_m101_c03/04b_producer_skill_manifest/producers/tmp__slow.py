exec(open('/tmp/graph.py').read().split('F,Y,rates=amps(q)')[0]); np.set_printoptions(precision=4,suppress=True,linewidth=150)
a=panel_fit(u,p['v1'],1,'orth')[0](z); b=panel_fit(u,p['v2'],2,'orth')[0](z); dot=(a*b).sum(1); ids=np.arange(len(y)); F,Y0,rates=amps(q); Y=np.c_[y,Y0];
D=aff(np.c_[a,b,dot]); M=(a[:,:,None]*b[:,None,:]).reshape(-1,16); bb=np.c_[b*b,pair(b)]; T=np.c_[np.ones(len(z)),b]; H=np.c_[np.ones(len(z)),a]; J=np.c_[np.ones(len(z)),z+.45*z*z]; G=np.c_[np.ones(len(z)),z];
folds=[]
for va in [ids%6==k for k in range(6)]+[ids>=56,np.linalg.norm(z,axis=1)>=np.sort(np.linalg.norm(z,axis=1))[-14],abs(Y[:,2])>=np.sort(abs(Y[:,2]))[-12]]:
 tr=~va; f=-np.expm1(-t[:,None]*[1.8,.4]); amp=ls(f,q[tr].T).T;folds.append((tr,va,f,amp))
def lin(m,tr,va,tar,reg=0):
 return m[va]@(ls(m[tr],tar) if not reg else ls(np.r_[m[tr],np.sqrt(reg)*np.eye(m.shape[1])],np.r_[tar,np.zeros(m.shape[1])]))
def model(tr,va,tar,name,design):
 if name=='linear':return lin(design,tr,va,tar,.0003)
 if name=='tanh_add':
  X=design; start=ls(X[tr],tar); L=D[tr]; U=D[va]
  def feat(rows,k):return np.c_[rows,np.tanh(k[0]*X[tr]@k[1:])]
  def err(k):
   m=feat(L,k);return m@ls(m,tar)-tar
  ini=np.r_[.7,start/max(np.linalg.norm(start),1e-6)]; opt=least_squares(err,ini,max_nfev=160);return np.c_[U,np.tanh(opt.x[0]*X[va]@opt.x[1:])]@ls(feat(L,opt.x),tar)
 if name=='tanh_axis':
  X=design
  def feat(mask,k):return np.c_[D[mask],np.tanh(X[mask]*k)]
  def err(k):
   m=feat(tr,k);return m@ls(m,tar)-tar
  opt=least_squares(err,[1.]*X.shape[1],bounds=(.15,8.),max_nfev=150);return feat(va,opt.x)@ls(feat(tr,opt.x),tar)
 X=design
 def fun(k,v):
  h=v@k[:-2]; c,d=k[-2:]
  if name=='tanh':return c*np.tanh(h)+d
  if name=='sin':return c*np.sin(h)+d
  if name=='rational':return c*h/(1+h*h)+d
  if name=='asinh':return c*np.arcsinh(h)+d
 def err(k):return fun(k,X[tr])-tar
 best=None
 for l in [.3,.7,1.2]:
  start=ls(X[tr],tar); c=max(abs(tar))*1.1;opt=least_squares(err,np.r_[start*l/c,c,0.],max_nfev=130)
  score=np.linalg.norm(opt.fun)
  if best is None or score<best[0]:best=score,opt.x
 return fun(best[1],X[va])
mods=[('linear',k,m) for k,m in {'D':D,'D+M':np.c_[D,M],'D+bb':np.c_[D,bb],'D+Mtri':np.c_[D,M,bb],'D+z':np.c_[D,z*z,pair(z)]}.items()]+[(n,k,m) for n in ('tanh','sin','rational','asinh','tanh_add') for k,m in [('a',H),('b',T),('z',G),('preb',J),('D',D)] if not(n=='tanh_add' and k=='D')]+[('tanh_axis',k,m) for k,m in [('a',a),('b',b),('z',z)]]
res=[]
for n,k,m in mods:
 E=[]
 for tr,va,f,amp in folds:
  h=lin(D,tr,va,amp[:,0],.00015); slow=model(tr,va,amp[:,1],n,m); err=np.abs(np.c_[h,slow]@f.T-q[va]); E.append(np.r_[np.mean(abs(slow-ls(f,q[va].T).T[:,1])),err.mean(0)])
 e=np.array(E);res.append((e[:6,1:].mean(),n,k,np.r_[e[:6,0].mean(),e[:6,1:].max(),e[6:,1:].max(1)]))
for row in sorted(res):print(row)
for co,tar in zip(ls(np.c_[D,M],Y).T,('scalar','fast','slow')):
 print(tar,'linear',co[:10],'mixed',co[10:].reshape(4,4))

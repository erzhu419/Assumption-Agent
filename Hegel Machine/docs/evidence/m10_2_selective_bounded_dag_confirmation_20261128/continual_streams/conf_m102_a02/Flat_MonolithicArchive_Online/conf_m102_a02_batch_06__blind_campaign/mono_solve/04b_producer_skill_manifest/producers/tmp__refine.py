exec(open('/tmp/discover.py').read().split('forms=')[0])
u,v=s['panel_b']['u'],s['panel_b']['v']; print('TELEM',s['telemetry_vec'],'TIMES',s['eval_times']); print('B range',np.min(u),np.max(u))
shapes={
'exp':lambda z,p,q:z*(-np.expm1(-p*abs(z)**q)),
'rat':lambda z,p,q:z*(1-1/(1+p*abs(z)**q)),
'rt':lambda z,p,q:z*(1-1/np.sqrt(1+p*abs(z)**q)),
'atan':lambda z,p,q:z*np.arctan(p*abs(z)**q),
'htan':lambda z,p,q:z*np.tanh(p*abs(z)**q),
'cos':lambda z,p,q:z*(1-np.cos(p*abs(z)**q)),
'exp2':lambda z,p,q:z**3*np.exp(-p*abs(z)**q),
'rat2':lambda z,p,q:z**3/(1+p*abs(z)**q),
'erf':lambda z,p,q:z*__import__('scipy.special',fromlist=['erf']).erf(p*abs(z)**q)}
res=[]
for name,fn in shapes.items():
 for mode in ['strict','aff','shift','sharedshift','cube','global','second']:
  def f(z,p,fn=fn,mode=mode):
   sh=p[2:] if mode in ('shift','sharedshift') else 0.; a=fn(z-sh,*p[:2]);
   base=np.ones((len(z),1)) if mode=='strict' else aff(z)
   if mode=='cube':base=np.c_[base,z**3]
   if mode=='global':base=np.c_[base,z*(z*z).mean(1)[:,None]]
   if mode=='second':base=np.c_[base,z*z]
   return np.c_[base,a]
  def err(p):
   A=f(u,p); return (A@ls(A,v)-v).ravel()
  ext=4 if mode=='shift' else int(mode=='sharedshift'); p=min((least_squares(err,[a,b]+[0]*ext,bounds=([.02,.3]+[-.6]*ext,[6,5]+[.6]*ext),max_nfev=90) for a,b in [(.7,2),(1.5,1),(.5,3),(2.5,2)]),key=lambda l:l.cost).x
  A=f(u,p); W=ls(A,v); E=A@W-v; lev=np.einsum('ij,ji->i',A,np.linalg.pinv(A)); res.append((np.mean(abs(E)/(1-lev[:,None])),np.mean(abs(E)),name,mode,p,lambda z,f=f,p=p,W=W:f(z,p)@W))
res.sort(key=lambda a:a[0]);
for r in res[:18]:print('op',r[:-1]);
Y=np.c_[s['train']['y'],s['train']['traj']]; U=s['panel_c']['u']; cf=lambda q:np.c_[aff(q),q*np.exp(-1.0223*abs(q)**.9585)]; df=lambda q:np.c_[aff(q),q*q,np.exp(-.7109*q*q)]; C=lambda q:cf(q)@ls(cf(U),s['panel_c']['v1']); D=lambda q:df(q)@ls(df(U),s['panel_c']['v2']);
for r in res[:5]:
 B=r[-1]; h=B(x); A=quad(h); ee=A@ls(A[:52],Y[:52])-Y; print('base',r[2:4],abs(ee[52:]).mean(0).round(4), 'hcorr',np.corrcoef(np.c_[Y[:,0],Y[:,-1],h].T)[:2].round(3));
 F=[]; names=[]
 def add(k,g):
  for j,p in enumerate(g.T):F.append(p);names.append(k+str(j))
 for zname,z in [('x',x),('h',h),('cx',C(x)),('dx',D(x)),('ch',C(h)),('dh',D(h)),('bcx',B(C(x))),('bdx',B(D(x)))]:
  add(zname,quad(z)[:,1:]); add(zname+'cube',z**3); add(zname+'tanh',np.tanh(z)); add(zname+'rat',z/(1+z*z))
 for xn,z in [('x',x),('cx',C(x)),('dx',D(x)),('ch',C(h)),('dh',D(h))]:
  add('h*'+xn,(h[:,:,None]*z[:,None,:]).reshape(len(x),-1))
 FF=np.asarray(F).T; R=FF-A@ls(A[:52],FF[:52]); E=-ee; scores=(R[:52].T@E[:52])/np.maximum(1e-9,np.linalg.norm(R[:52],axis=0))[:,None];
 for col in [0,-1]:
  ids=np.argsort(abs(scores[:,col]))[-8:][::-1]; print('resid features',col,[(names[j],round(scores[j,col],4),round(np.corrcoef(R[52:,j],E[52:,col])[0,1],2)) for j in ids]);
print('y/state relation',np.corrcoef(Y.T)[0].round(5)); print('trajectory sums',Y.mean(0).round(4))

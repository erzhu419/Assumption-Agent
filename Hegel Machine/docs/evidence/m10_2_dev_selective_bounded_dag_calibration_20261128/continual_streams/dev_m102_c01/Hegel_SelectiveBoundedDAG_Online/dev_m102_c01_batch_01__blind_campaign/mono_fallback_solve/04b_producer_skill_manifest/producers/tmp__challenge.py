exec(open('/tmp/models.py').read().split('rows=[]')[0])
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from itertools import combinations
print('y corr x/traj',np.corrcoef(np.c_[x,y,T[:,-1]].T)[4]); print('y lin',fit(design(x),y));print('traj lin',fit(design(x),T[:,-1]))
# Separate nonlinear primitives instead of requiring every row to pass through a single operator.
Z=[base[j] for j in ['X','A-','A+','BX','BA-','BA+']]
F=[np.ones((n,1)),x]
for z0 in Z:
 for k0 in [.5,1.,1.6,2.5]:
  F.extend([np.tanh(k0*z0),np.sin(k0*z0),z0*z0/(1+.165*z0*z0),poly(z0,2)[:,4:]])
for key in base:
 if key[0]=='C': F.extend([base[key],poly(base[key],2)[:,4:]])
F=np.column_stack(F); F=F[:,np.r_[True,np.std(F[:,1:],axis=0)>1e-10]]
for lam in [.003,.01,.03,.10,.30,1.]:
 K=F@F.T; out=check(K,lam); print('library',lam,out[0],out.mean(0),out.max(0))
for coord in ['X','BX','BA-','BA+']:
 z0=base[coord]; dd=(z0[:,None]-z0[None,:])**2; L=x@x.T; D=poly(x,2)[:,4:]; P=D@D.T
 for typ in ['iso','ard','ardq','res']:
  def make(h):
   w=np.exp(h); ds=dd@w[:4]; K=w[4]*np.exp(-ds)+w[5]*L+1.;
   if typ=='ardq' or typ=='res': K+=w[6]*P
   if typ=='res': K+=w[7]*(poly(z0,3)[:,14:]@poly(z0,3)[:,14:].T)
   return K
  npar={'iso':6,'ard':6,'ardq':7,'res':8}[typ]; outs=[]; hh=[]
  for j,l in splits:
   def unpack(h): return np.r_[np.repeat(h[0],4),h[1:]] if typ=='iso' else h
   def cost(h):
    w=unpack(h[:-1]); K=make(w)[np.ix_(j,j)]+np.exp(h[-1])*np.eye(len(j)); c=cho_factor(K); yy=y[j]; return .5*yy@cho_solve(c,yy)+np.log(c[0].diagonal()).sum()+.015*np.sum(h[:-1]**2)
   init=np.r_[[-1.5]*(1 if typ=='iso' else 4),0,0,[-3.]*(npar-6),-7.]
   opt=minimize(cost,init,bounds=[(-8,3)]*(len(init)-1)+[(-12,-2)],method='L-BFGS-B'); h=unpack(opt.x[:-1]); K=make(h); p=K[np.ix_(l,j)]@solve(K[np.ix_(j,j)]+np.exp(opt.x[-1])*np.eye(len(j)),y[j],assume_a='pos'); outs.append(np.mean(abs(p-y[l])));hh.append(np.exp(np.r_[h,opt.x[-1]]))
  print(coord,typ,np.round([outs[0],np.mean(outs),np.max(outs)],5),np.round(hh[0],5))
# Fit a separable tanh plus low-order interaction on the reference panel.
for mode in ['direct','odd','cross','skip','diag','atan','joint']:
 u=s['panel_c']['u']; v=s['panel_c']['v1']; ii,jj=np.triu_indices(4,1)
 def ff(h):
  k=np.exp(h[:4]); w=u*k
  g=np.arctan(w) if mode=='atan' else np.tanh(w)
  if mode=='joint': return design(np.tanh(w+h[4:8]))
  extra={'direct':[], 'odd':[u**3], 'cross':[u[:,ii]*u[:,jj]], 'skip':[u], 'diag':[u*u], 'atan':[]}[mode]
  return np.column_stack([design(g)]+extra)
 def fun(h): G=ff(h); return (G@fit(G,v)-v).ravel()
 o=least_squares(fun,np.r_[np.zeros(4)+.45,np.zeros(4) if mode=='joint' else []],bounds=(-2,2)); G=ff(o.x); Q=np.linalg.qr(G)[0]; print('panel',mode,np.std(o.fun),np.std(o.fun.reshape(-1,4)/(1-(Q*Q).sum(1)[:,None])),o.x)

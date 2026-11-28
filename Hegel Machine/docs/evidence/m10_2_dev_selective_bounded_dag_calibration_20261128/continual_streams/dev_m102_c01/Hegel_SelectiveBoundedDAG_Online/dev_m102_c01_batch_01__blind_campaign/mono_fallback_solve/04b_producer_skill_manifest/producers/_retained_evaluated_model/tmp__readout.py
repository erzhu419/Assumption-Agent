exec(open('/tmp/models.py').read().split('rows=[]')[0])
from scipy.optimize import least_squares
print('ranges',y.min(),y.max(),T[:,-1].min(),T[:,-1].max(),s['train']['summary'])
# Compare the time profile across amplitude bins and test an amplitude-dependent rate.
for jj in np.array_split(np.argsort(T[:,-1]),6):
 c=fit(T[jj,-1,None],T[jj]).ravel();print('time bin',T[jj,-1].mean(),c)
for typ in ('rat','exp','power','exp+rate','rat+rate'):
 def recur(h,tr):
  mag=tr[:,-1,None]; k=np.exp(h[0]+h[1]*mag+h[2]*mag*mag) if '+' in typ else np.exp(h[0]);
  if typ=='power': C=-np.expm1(-k*t**np.exp(h[1]));
  elif typ.startswith('rat'): C=t/(t+k);
  else:C=-np.expm1(-k*t)
  return tr[:,-1,None]*(C/C[..., -1:])
 o=least_squares(lambda h:(recur(h,T)-T).ravel(),[0.,0.,0.],bounds=(-3,3)); print('time fit',typ,o.x,np.mean(abs(o.fun)))
# Profile candidate saturating readouts with the saturation scale fit inside each fold.
for dest in [y,T[:,-1]]:
 cand=[]
 for name in ['X','A-','A+','BX','BA-','BA+']:
  z=base[name]; f=[design(z),np.c_[design(z),poly(z,2)[:,4:]]]
  i,j=np.triu_indices(4,1)
  for k in [.8,1.25,1.65]:
   f.append(np.c_[design(z),np.sin(k*z),np.cos(k*z),z[:,i]*z[:,j]])
  for m,F in enumerate(f):
   for sat in ['tanh','atan','asinh']:
    out=[]; pars=[]
    for a0,b0 in splits[:4]:
     a1=dest[a0]; amp=np.max(abs(a1));
     def ff(h,q):
      scale=amp*(1+np.exp(h[0])); p=q@h[1:]; return scale*({'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh}[sat])(p)
     starts=np.r_[0.,fit(F[a0],a1/(2*amp))]; starts=np.clip(starts,-20,20)
     obj=lambda h:ff(h,F[a0])-a1
     opt=least_squares(obj,starts,bounds=(np.r_[-6.,np.full(F.shape[1],-30.)],np.r_[4.,np.full(F.shape[1],30.)]),max_nfev=120);out.append(np.mean(abs(ff(opt.x,F[b0])-dest[b0]))); pars.append(np.mean(abs(opt.fun)))
    cand.append((np.mean(out),out[0],max(out),np.mean(pars),name,m,sat))
 print('readouts',sorted(cand)[:8])

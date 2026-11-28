exec(open('/tmp/probe.py').read().split('for g,ss in')[0])
p,h,_=sd
q=th[2]+m._features(di,di)[0]@th[3:]
y=Y(di); n=len(q)
def basis(tau): return np.vstack((np.r_[1/tau,1],np.c_[-np.expm1(-t[:,None]/tau),t]))
D=basis(tau); fixed=h[:,None]*(D[:,:2]@th[:2]); w=np.r_[0.45,np.ones(len(t))]; target=y-fixed
print('diagnostic q range',q.min(),q.max())
mods={'native':D,'native_no_linear':D[:,:2],'unit_initial_rate':D[:,:2]-D[:,2,None]/tau,'single_exp':None}
for name,T in mods.items():
 offset=q[:,None]*D[:,2] if name=='unit_initial_rate' else np.zeros_like(y)
 def run(idx):
  if T is None:
   def pred(v):
    s=np.exp(v[0]); return q[:,None]*v[1]*np.r_[1.,-s*np.expm1(-t/s)]
   fit=least_squares(lambda v:((target-offset-pred(v))[idx]*w).ravel(),[0,1]); return pred(fit.x),fit.x
  A=q[:,None,None]*T; b=np.linalg.lstsq((A[idx]*w[None,:,None]).reshape(-1,T.shape[1]),((target-offset)[idx]*w).ravel(),rcond=None)[0]; return A@b,b
 pred,b=run(np.ones(n,dtype=bool)); print(name,b,'fit rmse',np.sqrt(np.mean((y-fixed-offset-pred)**2,axis=0))[[0,1,-1]])
 for lab_out in [np.arange(n),di['lab_id']]:
  E=[]
  for lab in np.unique(lab_out):
   idx=lab_out==lab; pred,_=run(~idx); E.append((y-fixed-offset-pred)[idx]);
  print('worst group rate / curve rmse',np.max([[np.sqrt(np.mean(e[:,0]**2)),np.sqrt(np.mean(e[:,1:]**2))] for e in E],axis=0))
print('projected common profile / basis',np.linalg.lstsq(D*w[:,None], ((q@target)/(q@q))*w,rcond=None)[0])
print('archive solver setup'); print(''.join(open(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]).readlines()[:62]))

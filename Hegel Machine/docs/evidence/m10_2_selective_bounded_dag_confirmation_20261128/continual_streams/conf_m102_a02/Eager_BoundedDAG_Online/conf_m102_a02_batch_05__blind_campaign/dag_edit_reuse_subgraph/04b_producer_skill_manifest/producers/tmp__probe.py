exec(open('/tmp/explore.py').read().split('for p in (')[0])
from scipy.optimize import least_squares
p=s['panel_a']; u=p['u']; v=p['v']; m=np.array([np.polyfit(u[:,j],v[:,j],1) for j in range(4)]); g=np.max(np.abs(v-u*m[:,0]-m[:,1]),axis=1)<.5; m=np.array([np.polyfit(u[g,j],v[g,j],1) for j in range(4)]); print('cal',m); z=(x-m[:,1])/m[:,0]; w=(s['test']['x']-m[:,1])/m[:,0]; print('normalized quantiles',np.quantile(z,[0,.1,.25,.5,.75,.9,1],axis=0)); print('test quantiles',np.quantile(w,[0,.1,.5,.9,1],axis=0)); print('calibrated rows',np.c_[z[:15],y[:15],t[:15,-1]]); print('endpoint residual cubic',np.c_[z,y,y-feat(z,3)@fit(z,y,3)][np.argsort(np.abs(y-feat(z,3)@fit(z,y,3)))[-12:]]); print('ridge',[(d,cv(feat(z,d),Y,.03)) for d in (2,3)]);
for p,k in ((s['panel_b'],'v'),(s['panel_c'],'v1'),(s['panel_c'],'v2')):
 for l,u in p.items():
  if k==l:continue
  F=feat(u,2); V=p[k]; B=fit(u,V,2); print('quadratic',l,'->',k, np.sqrt(np.mean((F@B-V)**2))); print(B.T)
  if l!='u':continue
  for name,f in [('sin',np.sin),('tanh',np.tanh),('softsign',lambda a:a/(1+abs(a))),('rational',lambda a:a/(1+a*a)),('cubic',lambda a:a+a**3),('quad',lambda a:a+a*a)]:
   A=feat(u,1); E=[]
   for j in range(4):
    aj=np.linalg.lstsq(A,V[:,j],rcond=None)[0]; r=least_squares(lambda b:b[0]+b[1]*f(A@b[2:])-V[:,j],np.r_[0,1,aj],max_nfev=500); E.append(np.sqrt(np.mean(r.fun**2)))
   print(name,np.round(E,5))

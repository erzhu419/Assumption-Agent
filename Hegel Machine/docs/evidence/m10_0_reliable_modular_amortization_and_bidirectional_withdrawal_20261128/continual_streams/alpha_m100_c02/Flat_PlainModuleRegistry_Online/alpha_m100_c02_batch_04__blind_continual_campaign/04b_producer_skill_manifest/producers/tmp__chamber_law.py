from chamber_probe import *
from scipy.linalg import lstsq
q,th=fit(); p,h=load(c); g=gap(c,q); d0=xc@th[2:]; xlocal=features(a,c)[1]; dl=xlocal@th[2:]; btime=np.r_[t,1.]; prev=h[:,None]-g
print('tau/theta',q,th,'diag missing',np.isnan(c['coupon_features_raw']).sum(0),'drift delta pooled train-test vs train-diag',np.round(d0-dl,4))
for name,V in [('reset drift',np.c_[h,d0]),('reset local drift',np.c_[h,dl]),('reset free drift',np.c_[h,xc]),('history drift',np.c_[h,d0,prev]),('history free drift',np.c_[h,xc,prev]),('nonlinear reset',np.c_[h,d0,h*d0,d0**2,h**2]),('nonlinear history',np.c_[h,d0,prev,h*d0,d0**2,h**2])]:
 def run(m,test,free,show=False):
  yy=yc[m]; U=V[m];
  def calc(logtau,final=False):
   B=np.c_[basis(np.exp(logtau)),btime]; A=np.einsum('ni,tj->ntij',U,B).reshape(len(U),len(unit),-1); w=A*unit[None,:,None]; co=lstsq(w.reshape(-1,w.shape[-1]),(yy*unit).ravel())[0].reshape(U.shape[1],-1)
   if final:return V[test]@co@B.T,co
   return ((U@co@B.T-yy)*unit).ravel()
  opt=least_squares(calc,np.log(q),bounds=(np.log([.07,.9]),np.log([1.1,24.])),max_nfev=70) if free else None
  qp=opt.x if free else np.log(q); pred,co=calc(qp,True)
  if show: print('   ',name,'free',free,'taus',np.round(np.exp(qp),5),'in',err(yc[test],pred),'coeff',np.round(co,4).tolist())
  return pred
 for free in (False,True):
  run(np.ones(len(h),bool),np.ones(len(h),bool),free,True)
  yh=np.empty_like(yc); errors=[]
  for lab in np.unique(c['lab_id']):
   test=c['lab_id']==lab; yh[test]=run(~test,test,free); errors.append(err(yc[test],yh[test])[:2])
  print('    OOF',err(yc,yh)[:2],'worst lab',np.max(errors,0))
print('pointwise fit of hold and drift, incl initial rate:')
V=np.c_[h,d0]; co=np.linalg.lstsq(V,yc,rcond=None)[0]; print(np.round(co,5)); print('rmse by time',np.round(np.sqrt(np.mean((V@co-yc)**2,axis=0)),5)); print('relative to reset material',np.round(co[0]-basis(q)@th[:2],5)); print('diag residual std by row',np.round(np.sqrt(np.mean((V@co-yc)**2,axis=1)),4))

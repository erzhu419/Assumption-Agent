exec(open('/tmp/chamber_probe.py').read().split('for drift in')[0])
D=np.r_[t,1.]
def temp(q): return np.vstack([et(a) for a in q]+[D])
A=temp(tau); coef=np.linalg.lstsq(A.T,y.T,rcond=None)[0].T
G=c[:,0,:2]/(-np.expm1(-t[0]/tau)); B=np.column_stack((G*th[:2],dr)); diff=coef-B
print('amplitude medians baseline/diff',np.median(B,0),np.median(diff,0)); print('first amplitudes [hold, gaps, drift, A0,A1,Adrift]',np.round(np.column_stack((h,G,dr,coef)),4))
from itertools import combinations
from scipy.linalg import qr
labs=d['lab_id']; z=h[:,None]-G
for label,F in [('basic',np.column_stack((np.ones(nd),h,z,dr))),('history',np.column_stack((np.ones(nd),h,past,dr))),('history+x',np.column_stack((np.ones(nd),h,past,xd))),('simple nonlinear',np.column_stack((np.ones(nd),h,z,dr,h*h,z*z,h[:,None]*z,h*dr,z*dr[:,None],dr*dr))),('differences',np.column_stack((np.ones(nd),h,G,dr,h*dr,h[:,None]*G,G*dr[:,None],G**2,h**2,dr**2)))]:
 for target,T in [('total',coef),('delta',diff)]:
  C=np.linalg.lstsq(F,T,rcond=None)[0]; o=np.zeros_like(T)
  for l in np.unique(labs):
   m=labs!=l; o[~m]=F[~m]@np.linalg.lstsq(F[m],T[m],rcond=None)[0]
  e=(o-T)@A; print(label,target,'in residual',np.sqrt(np.mean(((F@C-T)@A)**2)),'LOLO RMSE',np.sqrt(np.mean(e**2)),'rate MAE',np.mean(abs(e[:,-1]))); print('coeffs',np.round(C.T,4))
# Direct joint fit: load-memory amplitudes projected for each candidate chamber time pair.
def proj(q,F,Y=y):
 V=temp(q); C=np.linalg.lstsq(F,Y,rcond=None)[0]; para=np.linalg.lstsq(V.T,C.T,rcond=None)[0].T; return F@para@V-Y,para
for label,F in [('gaps',np.column_stack((np.ones(nd),h,z,dr))),('history',np.column_stack((np.ones(nd),h,past,dr))),('history+x',np.column_stack((np.ones(nd),h,past,xd)))]:
 q=np.exp(least_squares(lambda w:proj(np.exp(w),F)[0].ravel(),np.log(tau),bounds=(np.log([.1,1]),np.log([1.5,30]))).x); e,beta=proj(q,F); print('joint',label,q,np.sqrt(np.mean(e**2)),beta.T.round(4))

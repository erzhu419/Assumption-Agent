from m99_probe import *
from itertools import product
from scipy.optimize import minimize_scalar
G=lambda q: pd @ (np.exp(-(np.cumsum(dt[::-1])[::-1]-dt)[:,None]/np.atleast_1d(q))*-np.expm1(-dt[:,None]/np.atleast_1d(q)))
D=beta[2]+xd@beta[3:]
F={'1':np.ones((len(hd),1)),'h':hd[:,None],'z0':G(tau)[:,:1],'z1':G(tau)[:,1:],'d':D[:,None],'x':xd,'p':pd,'mean':pd.mean(1)[:,None],'last':pd[:,-1:]}
O=np.c_[Y,R]
def solve(spec, ix, ts=None, plus=False, params=None):
 f=[np.hstack([F[k] for k in s.split('+')]) for s in spec]; ns=np.cumsum([0]+[a.shape[1] for a in f]);
 def proj(logt,ret=False):
  ta=np.exp(logt); K=np.r_[np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1][None,:]]
  aa=np.concatenate([(a[:,None,:]*K[None,:,j,None]) for j,a in enumerate(f)],axis=-1)
  B=np.linalg.lstsq(aa[ix].reshape(-1,ns[-1]),O[ix].ravel(),rcond=None)[0]; E=aa@B-O
  return (ta,B,E) if ret else E[ix].ravel()
 if ts is not None:return proj(np.log(ts),True)
 opt=least_squares(proj,np.log([.63,1.15]),bounds=(np.log([.25,.75]),np.log([.75,8])),max_nfev=100)
 return proj(opt.x,True)
def score(E):return np.r_[np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2))]
if __name__=='__main__':
 all_=np.arange(len(hd)); lab=dg['lab_id'];
 for spec in [('1+h+z0+d','1+h+z1+d','1+h+d'),('1+h+z0','1+h+z1+d','1+h+d'),('1+h+z0+z1+d',)*3,('1+h+z0+z1+x',)*3,('1+h+p+d',)*3,('1+h+p+x',)*3,('1+h+z0','1+h+z1+d','h'),('h+z0','h+z1+d','h'),('1+h+z0','1+h+d','h'),('1+h+z0','1+h+z0+d','h'),('1+h+mean','1+h+mean+d','h')]:
  ta,b,e=solve(spec,all_); cv=[]
  for l in np.unique(lab):
   _,_,el=solve(spec,lab!=l);cv.append(score(el[lab==l]));
  print(spec,'tau',ta.round(5),'fit',score(e).round(5),'CV',np.array(cv).round(5).tolist(),'beta',b.round(5));
 print('miss',np.isnan(dg['coupon_features_raw']).astype(int).T.tolist());print('features D',D.round(4));print('lab',lab)

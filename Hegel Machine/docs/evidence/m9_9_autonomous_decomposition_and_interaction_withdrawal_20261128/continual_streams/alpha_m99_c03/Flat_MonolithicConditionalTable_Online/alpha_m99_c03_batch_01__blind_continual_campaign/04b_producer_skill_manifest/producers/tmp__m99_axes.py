from m99_inputs import *
from scipy.optimize import least_squares
D=np.vstack((E,er)); O=np.c_[y,yr]; n=len(O); p,h,_=cg; labs=g['lab_id'];
xx=m._features(g,g)[-n:]; sc=xx@th[3:]; off=np.c_[h[:,None]*th[:2],np.zeros(n)]
R=O-off@D.T
for j in [0,5,11,12]:
 F=np.c_[np.ones(n),xx]; A=np.linalg.lstsq(F,R[:,j],rcond=None)[0]; k=np.linalg.lstsq(np.c_[np.ones(n),sc],R[:,j],rcond=None)[0]; e=R[:,j]-k[0]-k[1]*sc
 print('time',j,'score multiplier',k.round(5),'full feature multipliers', (A[1:]/th[3:]).round(4),'mean residual by lab',[round(e[labs==l].mean(),5) for l in np.unique(labs)])
print('reference scaling: lab, diag mean/std, test mean/std')
for l in np.unique(labs):
 print(l,*[(round(np.median(b['ref_coupons'][b['lab_id']==l].mean(0)),4),round(np.median(b['ref_coupons'][b['lab_id']==l].std(0)),4)) for b in (g,te)])
print('raw feature sample: missing counts train test diag',*[np.isnan(b['coupon_features_raw']).sum(0).tolist() for b in (tr,te,g)])

import warnings
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')
from assay_dev import *
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score

# Compare low-variance priors, without using query outcomes to fit or select.
def fit(z,s,y,l2,shared,trend):
    X=np.column_stack((np.ones(len(y)),z,s==1,s==2))
    P=np.diag(np.r_[0.,np.full(z.shape[1],l2),.25,.25]); v=np.r_[np.zeros(1+z.shape[1]),2.,-1.]; P+=trend*np.outer(v,v)
    def solve(P):
        def fg(b):
            eta=X@b; pb=P@b
            return np.sum(np.logaddexp(0,eta)-y*eta)+.5*b@pb,X.T@(expit(eta)-y)+pb
        return minimize(fg,np.zeros(X.shape[1]),jac=True,method='BFGS',options={'gtol':1e-7}).x
    b=solve(P)
    if not shared:return b
    sign=np.sign(b[1:-2]); P[1:-2,1:-2]+=shared*(np.eye(len(sign))-np.outer(sign,sign)/len(sign))
    return solve(P)
def pred(b,z,s): return expit(b[0]+z@b[1:-2]+b[-2]*(s==1)+b[-1]*(s==2))
def score(y,p):return np.array([roc_auc_score(y,p),brier_score_loss(y,p),balanced_accuracy_score(y,p>=.5)])
configs=[('raw',1,0,0),(.2,1,0,0),(.2,3,0,0),(.2,1,0,1),(.2,1,0,5),(.2,1,5,0),(.2,1,5,1),(.2,1,12,1),(.45,1,0,0),(.45,1,5,1)]
for i,e in enumerate(cohorts):
 d,g=e['dataset'],e['gt']; r,t,q=(d[k] for k in ('reference_controls','train','query')); tc,qc=t['center_id'],q['center_id']; st,sq=t['stratum'],q['stratum']; y,yq=t['y'],g['y_query']
 print('cohort',i)
 for a,l2,shared,trend in configs:
  z,zq=(np.nan_to_num(t['X']),np.nan_to_num(q['X'])) if a=='raw' else (align(r,t,a),align(r,q,a))
  cv=[]
  for c in np.unique(tc):
   j=tc!=c; b=fit(z[j],st[j],y[j],l2,shared,trend); cv.append(score(y[~j],pred(b,z[~j],st[~j])))
  cv=np.asarray(cv); b=fit(z,st,y,l2,shared,trend); p=pred(b,zq,sq); qs=np.array([score(yq[qc==c],p[qc==c]) for c in np.unique(qc)])
  print((a,l2,shared,trend),'CVmean',cv.mean(0).round(4),'CVworst',np.array([cv[:,0].min(),cv[:,1].max(),cv[:,2].min()]).round(4),'query A/B',qs[:,:2].ravel().round(4),'weights',b[1:-2].round(2),'stratum',b[-2:].round(2))

import warnings
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')
from assay_dev import *
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, log_loss

def fit(z,s,c,y,C,center_sd):
    u,gi=np.unique(c,return_inverse=True)
    X=np.column_stack((np.ones(len(y)),z,s[:,None]==[1,2],np.eye(len(u))[gi]))
    lam=np.r_[0,np.repeat(1/C,z.shape[1]+2),np.repeat(1/center_sd**2,len(u))]
    def fg(b):
        eta=X@b
        return np.sum(np.logaddexp(0,eta)-y*eta)+.5*np.dot(lam*b,b),X.T@(expit(eta)-y)+lam*b
    return minimize(fg,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'ftol':1e-12}).x

def pr(b,z,s): return expit(b[0]+design(z,s)@b[1:11])
def metrics(y,p):return np.array([roc_auc_score(y,p),brier_score_loss(y,p),balanced_accuracy_score(y,p>=.5)])
for e,entry in enumerate(cohorts):
 d,g=entry['dataset'],entry['gt']; r,t,q=[d[k] for k in ('reference_controls','train','query')]; tc,qc=t['center_id'],q['center_id']; y=t['y']; st,sq=t['stratum'],q['stratum']; u=np.unique(tc)
 print('cohort',e,'oracle overall',metrics(g['y_query'],g['prob_query']))
 for a in [0.,.2,.6,1.]:
  z,zq=align(r,t,a),align(r,q,a)
  for C in [.1,.5,3.,20.]:
   for sd in [.001,.3,2.]:
    cv=[]
    for k in u:
     i=tc!=k; b=fit(z[i],st[i],tc[i],y[i],C,sd); cv.append(metrics(y[~i],pr(b,z[~i],st[~i])))
    b=fit(z,st,tc,y,C,sd); p=pr(b,zq,sq); qs=np.array([metrics(g['y_query'][qc==c],p[qc==c]) for c in np.unique(qc)])
    cv=np.asarray(cv)
    print('a,C,sd',a,C,sd,'CV avg',cv.mean(0),'CV minA, maxB',cv[:,0].min().round(3),cv[:,1].max().round(3),'Q',qs[:,:2].ravel(),'QprobRMSE',np.mean((p-g['prob_query'])**2)**.5)
 z=align(r,t,.2)
 for c in u:
  ix=tc==c; b=fit(z[ix],st[ix],tc[ix],y[ix],3,.001); s=z[ix]@g['beta']; print('train',c,'mean/SD z',z[ix].mean(0),z[ix].std(0),'strata',np.bincount(st[ix]),'fixed oracle slope + strata intercepts only',[(float(y[ix][st[ix]==j].mean()),float(s[st[ix]==j].mean())) for j in range(3)],'fit',b[:11])

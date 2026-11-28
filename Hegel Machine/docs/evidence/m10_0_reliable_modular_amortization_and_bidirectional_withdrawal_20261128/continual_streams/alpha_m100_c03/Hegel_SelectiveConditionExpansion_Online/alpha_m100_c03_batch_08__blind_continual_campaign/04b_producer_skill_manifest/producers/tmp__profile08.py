exec(open('/tmp/probe08.py').read().split("print('intrinsic'")[0])
from scipy.optimize import least_squares
T=np.r_[t,1.]; E=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); D=E*tau-T[:,None]; ref=h[:,None]*(E@th[:2]); observed=np.c_[y(q),q['instant_rate_obs']]; rr=th[2]+_features(q,q)@th[3:]; rr=rr[:len(h)]; labs=q['lab_id']; groups=np.unique(labs); withdrawal=h<p[:,-1]

def curve(g): return T+D@np.array(g)
def project(g,mask):
 b=curve(g); Y=observed[mask]-ref[mask]; return Y@b/(b@b), b

def fit(mask):
 def calc(g):
  r,b=project(g,mask); return (ref[mask]+r[:,None]*b-observed[mask]).ravel()
 return least_squares(calc,[.2,.55],bounds=(0,1),loss='soft_l1',f_scale=.004).x

def report(label, pred):
 e=pred-observed
 print(label,'overall curve/rate',np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),'lab and withdrawal curve/rate/final',[(int(m.sum()),*[round(np.sqrt(np.mean(a*a)),6) for a in (e[m,:-1],e[m,-1],e[m,-2])]) for m in [*(labs==i for i in groups),withdrawal,~withdrawal]])

G=fit(np.ones(len(h),bool)); print('profiled routing',G,'linear',1-sum(G)); rout=np.empty_like(observed)
for i in groups:
 m=labs==i; gi=fit(~m); rout[m]=ref[m]+rr[m,None]*curve(gi); print('omit lab',i,gi)
report('LOLO amplitudes chosen without holdout responses',rout)
report('fixed (.2,.55)',ref+rr[:,None]*curve([.2,.55])); rlat,b=project(G,np.ones(len(h),bool)); print('profile residual by horizon',np.sqrt(np.mean((ref+rlat[:,None]*b-observed)**2,axis=0))); print('missing counts / discrepancy',np.c_[np.isnan(q['coupon_features_raw']).sum(1),rr,rlat,rlat-rr].round(5)); print('average by lab',[(i,np.mean((rlat-rr)[labs==i])) for i in groups]); print('test missing counts',np.bincount(np.isnan(te['coupon_features_raw']).sum(1)))

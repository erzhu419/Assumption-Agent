exec(open('/tmp/chamber_memory.py').read().split("print('zero-drift")[0])
from itertools import product
aux=te['aux_chamber_telemetry']; print('aux shape',aux.shape,'mean',aux.mean(0).round(4),'std',aux.std(0).round(4))
print('assets',[(k,str(v)[:160]) for k,v in __import__('json').load(open('/holdout_task/public_asset_index.json')).items()])
empty={k:dg[k][:0] for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}
xd=m._features(dg,empty); r=th[2]+xd@th[3:]; T=np.r_[t,1.]; miss=np.isnan(dg['coupon_features_raw']).any(1); full=~miss

def rms(e): return np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))
def run(fitq, drift, free_hold, select, intrinsic=taus, weights=th):
 r=weights[2]+xd@weights[3:]
 def fit(j):
  def eval(q, ret=False):
   B=basis(q); base=h[:,None]*(B@weights[:2]); F=[r[:,None]*B[:,k] for k in range(2)];
   if drift: F.append(r[:,None]*T)
   if free_hold: F.extend([h[:,None]*B[:,k] for k in range(2)])
   F=np.stack(F,axis=-1); X=F[j].reshape(-1,F.shape[-1]); v=np.linalg.lstsq(X,(y-base)[j].ravel(),rcond=None)[0]; e=base+F@v-y
   return (q,v,e) if ret else e[j].ravel()
  q=least_squares(eval,intrinsic,bounds=([.5,2],[1,7])).x if fitq else intrinsic
  return eval(q,True)
 q,v,e=fit(select); folds=[fit(select&(lab!=l))[2][lab==l] for l in np.unique(lab)]
 print(fitq,drift,free_hold,'n',sum(select),'q',q.round(5),'coeff',v.round(5),'complete/missing',np.round([rms(e[full]),rms(e[miss])],5),'worst lab',np.max([rms(e) for e in folds],0).round(5),'worst by output',max(np.sqrt(np.mean(e[:,:-1]**2,axis=0))).round(5))
for fitq,drift,free_hold in product((False,True),repeat=3): run(fitq,drift,free_hold,np.ones(N,bool))
run(True,True,False,full)
print('missing detail lab/raw/local standardized',np.c_[lab[miss],dg['coupon_features_raw'][miss],xd[miss],r[miss]].round(5))
for name,g in [('train',tr),('test',te),('diag',dg)]:
 print(name,'stratum counts',np.unique(np.c_[g['lab_id'],g['stratum']],axis=0,return_counts=True)[1]); print(name,'local means',m._features(g,empty).mean(0).round(5))
for name,xtr in [('joint',x[:n]),('local',m._features(tr,empty)),('train+diag',m._features(tr,dg)[:n])]:
 q,w=m._fit(m._coupon_inputs(tr,xtr,b),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('intrinsic',name,q.round(5),w.round(5)); run(True,True,False,np.ones(N,bool),q,w)

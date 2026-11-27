from rheo_probe import *

def cut(q,m): return {k:v[m] for k,v in q.items()}
def cat(*parts): return {k:np.concatenate([q[k] for q in parts]) for k in parts[0] if all(k in q for q in parts)}
def fit2(q,qr,scale,prior=None,shrink=0.):
    y,r=target(q); target_flat=np.r_[y.ravel(),.7*r];
    def matrix(logtau):
        A,B=designs(q,np.exp(logtau),'z',qr); return np.r_[A.reshape(-1,7),.7*B]
    def fun(par):
        er=matrix(par[:2])@par[2:]-target_flat
        if shrink: er=np.r_[er,shrink*(par[2:]-prior)]
        return er
    best=None
    for init in [(.45,2),(.85,6)]:
        lt=np.log(init); w=np.linalg.lstsq(matrix(lt),target_flat,rcond=None)[0]; o=least_squares(fun,np.r_[lt,w],bounds=([np.log(.05001)]*2+[-np.inf]*7,[np.log(24.999)]*2+[np.inf]*7),loss='soft_l1',f_scale=scale,max_nfev=95)
        if best is None or o.cost<best.cost: best=o
    ix=np.argsort(best.x[:2]); return np.exp(best.x[:2][ix]),np.r_[best.x[2:4][ix],best.x[4:]]

def errs(q,f,ref):
    tau,w=f; A,B=designs(q,tau,'z',ref); y,r=target(q); return A@w-y,B@w-r

def summ(e,r):
    return np.round([np.median(abs(e)),np.quantile(abs(e),.9),np.quantile(abs(e[:,:3]),.9),np.quantile(abs(e[:,-3:]),.9),np.median(abs(r)),np.quantile(abs(r),.9)],5)

if __name__=='__main__':
 print('rows missing', [(len(q['lab_id']),np.sum(np.isnan(q['coupon_features_raw']),axis=0).tolist()) for q in (tr,ch,te)])
 for qlabel,q in [('bench',tr),('pilot',ch)]:
  for label,ref in [('own',q),('bench',tr),('pilot',ch),('all',cat(tr,ch,te))]:
   for sc in (.002,.009):
    f=fit2(q,ref,sc); e,r=errs(q,f,ref); complete=np.isfinite(q['coupon_features_raw']).all(1); print('in',qlabel,label,sc,np.round(f[0],4),np.round(f[1],5),summ(e,r),'complete',summ(e[complete],r[complete]))
 bench=fit2(tr,tr,.009); allref=cat(tr,ch,te)
 for label, qr, er, qfit, sc in [('bench-only',tr,ch,tr,.009),('pilot-own',ch,ch,ch,.003),('pilot-bench-ref',tr,tr,ch,.003),('pilot-pooled-ref',allref,allref,ch,.003),('pooled-labels',allref,allref,None,.006)]:
  E=[]; R=[]; W=[]; T=[]
  for lab in np.unique(ch['lab_id']):
   train=cut(ch,ch['lab_id']!=lab); test=cut(ch,ch['lab_id']==lab); f=bench if qfit is tr else fit2(train if qfit is ch else cat(tr,train),qr,sc); e,r=errs(test,f,er); E.extend(e);R.extend(r); W.append(summ(e,r));T.append(f[0])
  print('LOLO pilot',label,'pooled',summ(np.array(E),np.array(R)),'worst',np.max(W,axis=0),'taus',np.round(T,3))
 for s in ('train','chamber_verification_trials'):
  q=p[s]; y,r=target(q); print('ranges',s,'stress', [np.round(np.percentile(v,[0,10,50,90,100]),3) for v in prep(q)[1:]],'zero',np.round(np.percentile(q['strain_0'],[0,25,50,75,100]),4));

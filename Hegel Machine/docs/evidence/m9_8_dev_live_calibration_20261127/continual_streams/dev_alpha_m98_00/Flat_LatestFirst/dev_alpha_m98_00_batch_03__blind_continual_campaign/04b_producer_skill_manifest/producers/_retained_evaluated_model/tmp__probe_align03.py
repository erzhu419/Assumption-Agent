exec(open('/tmp/probe03.py').read().split('err(\'intrinsic-time')[0])
from itertools import combinations
print('diagnostic overlap with deployment by hold raw',np.min(np.linalg.norm(a['hold_channel_raw'][:,None]-te['hold_channel_raw'],axis=2),axis=1).round(4))
D,R=m._design(tau,dt,t,co)
z=h[:,None]-D[:,0,:2]/F[0,:2]; base=np.c_[D@b,R@b]; branch=(h[:,None]-z)*b[:2]; resid=y-branch@F[:,:2].T
u,s,vt=np.linalg.svd(resid,full_matrices=False); print('fixed-branch residual singular values',s.round(5));print('residual projections fast slow steady',np.linalg.lstsq(F,(u[:,0]*s[0])@np.atleast_2d(vt[0]) if False else vt[0],rcond=None)[0]);print('base rate residual',np.c_[labs,v,y[:,-1]-R@b].round(4));
opts={('train','test'):v}
groups={'train':tr,'test':te,'diag':a}
for ns in [('train',),('test',),('diag',),('train','diag'),('test','diag'),('train','test','diag')]:
 s={k:np.concatenate([groups[n][k] for n in ns]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}; f=a['coupon_features_raw'].copy(); g=s['coupon_features_raw'].copy()
 for j in np.unique(labs):
  rr=s['ref_coupons'][s['lab_id']==j];mu=np.median(rr.mean(0));sd=np.median(rr.std(0));f[labs==j]=(f[labs==j]-mu)/sd;g[s['lab_id']==j]=(g[s['lab_id']==j]-mu)/sd
 _,ci,ct=np.unique(np.c_[s['lab_id'],s['stratum']],axis=0,return_inverse=True,return_counts=True);w=1/ct[ci,None];ff=np.nansum(g*w,axis=0)/np.sum(w*np.isfinite(g),axis=0);opts[ns]=b[2]+np.where(np.isnan(f),ff,f)@b[3:]

def report(name,pred):
 e=pred-y; print(name,'pooled',np.sqrt((e**2).mean(0))[[0,5,-2,-1]].round(5),'worst',np.max([np.r_[np.sqrt(np.mean(e[labs==j,:-1]**2)),np.sqrt(np.mean(e[labs==j,-1]**2))] for j in np.unique(labs)],axis=0).round(5))
for ns,w in opts.items():
 print('anchor',ns,'drift vs measured initial excess RMSE',np.sqrt(np.mean((w-resid[:,-1])**2)).round(4),'corr w inferred drift',np.corrcoef(w,u[:,0])[0,1].round(4));
 pool=np.stack([w[:,None]*F[:,k] for k in range(3)],axis=2);pred=0*y;pred2=0*y;cs=[]
 for j in np.unique(labs):
  keep=labs!=j;qa=pool[keep].reshape(-1,3);qy=resid[keep].ravel();c=np.linalg.lstsq(qa,qy,rcond=None)[0];cs.append(c);pred[~keep]=branch[~keep]@F[:,:2].T+pool[~keep]@c
  constr=pool[:,:,:2]-pool[:,:,2,None]/tau;lin=pool[:,:,2];c=np.linalg.lstsq(constr[keep].reshape(-1,2),(resid-lin)[keep].ravel(),rcond=None)[0];pred2[~keep]=branch[~keep]@F[:,:2].T+lin[~keep]+constr[~keep]@c
 print('coef by fold',np.round(cs,4));report('  free forcing',pred);report('  initial drift conserved',pred2)
print('memory + forcing regression with fixed hold gains')
w=opts[('train','test')];B=np.stack([-z[:,0,None]*F[:,0],-z[:,1,None]*F[:,1],*(w[:,None]*F[:,k] for k in range(3))],axis=2); y0=(h[:,None]*b[:2])@F[:,:2].T
for j in [-1,*np.unique(labs)]:
 keep=labs!=j;c=np.linalg.lstsq(B[keep].reshape(-1,5),(y-y0)[keep].ravel(),rcond=None)[0];print(j,c.round(5),np.sqrt(np.mean((y0+B@c-y)[~keep if j!=-1 else keep]**2,axis=0))[[0,5,-2,-1]].round(4))

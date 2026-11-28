exec(open('/tmp/probe.py').read().split("print('bridge RMS/max'")[0])
o=m['solve_batch'](d); tau=o['estimated_taus']; w=o['hereditary_weights']; print('tau,theta:',tau,w); print('bridge residual:',np.sqrt(np.mean(er**2)),np.max(abs(er)))
xt,xe=m['_features'](tr,te); xc1=m['_features'](tr,c)[1]; xc2=m['_features'](dict((k,np.concatenate((tr[k],te[k]))) for k in ['lab_id','stratum','ref_coupons','coupon_features_raw']),c)[1]
def frozen_features(c):
 keys=['lab_id','stratum','ref_coupons','coupon_features_raw']; p={k:np.concatenate([tr[k],te[k]]) for k in keys}; x=c['coupon_features_raw'].copy(); u=p['coupon_features_raw'].copy(); lab=p['lab_id'];
 for l in np.unique(lab):
  r=p['ref_coupons'][lab==l]; center=np.median(r.mean(0)); scale=np.median(r.std(0)); x[c['lab_id']==l]=(x[c['lab_id']==l]-center)/scale; u[lab==l]=(u[lab==l]-center)/scale
 _,ix,n=np.unique(np.c_[lab,p['stratum']],axis=0,return_inverse=True,return_counts=True); f=1/n[ix.ravel(),None]; mu=np.nansum(u*f,0)/np.sum(f*np.isfinite(u),0); return np.where(np.isfinite(x),x,mu)
xc3=frozen_features(c)
s=m['_prepare'](c,xc3,B); past,hold,f=s
H,R=m['_design'](np.log(tau),s,dt,t); g=R[:,:2]*tau; z=hold[:,None]-g
T=np.r_[np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1][None,:]]; Y=np.c_[c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]; A=Y@np.linalg.pinv(T).T
print('chamber drift vs each aligned base drift:')
for x in (xc1,xc2,xc3):
 F=np.c_[np.ones(len(x)),x]; drift=F@w[2:]; print(np.linalg.lstsq(drift[:,None],A[:,2],rcond=None)[0],np.sqrt(np.mean((A[:,2]-.035*drift)**2)),np.linalg.lstsq(F,A[:,2],rcond=None)[0]);
chan=lambda raw:raw@B[1:]+B[0]
P,Q=chan(c['past_channel_raw']),chan(c['hold_channel_raw']); K=np.exp(-np.r_[np.cumsum(dt[:0:-1])[::-1],0.][:,None]/tau)*-np.expm1(-dt[:,None]/tau); Z=np.einsum('nsc,sb->ncb',P,K); C=Q[:,:,None]-Z
one=np.ones(len(g)); lab=np.eye(4)[c['lab_id']]; D={ 'gap':g,'hold':hold[:,None], 'past':past,'hold_chan':Q,'past_last_chan':P[:,-1], 'all_chan_gap':C.reshape(len(g),-1),'zchan':Z.reshape(len(g),-1),'features':f[:,1:],'group':lab[:,1:],'gap2':g*g,'hold_gap':hold[:,None]*g,'abs_chan':np.abs(Q),'channel_spread':Q.std(axis=1)[:,None]};
for name, cand in [('own gap',None),('gap only',[]),('hold',['hold']),('past',['past']),('history',['hold','past']),('hold chan',['hold_chan']),('chan memory',['all_chan_gap']),('hold chan+memory',['hold_chan','all_chan_gap']),('features',['features']),('group',['group']),('polynomial',['hold','gap2','hold_gap']),('hold spread',['hold','channel_spread'])]:
 print(name,end=' ')
 for j in range(3):
  X=np.c_[one,g[:,[min(j,1)]]] if cand is None else np.column_stack([one,g]+[D[k] for k in cand]); target=A[:,j]; fit=np.linalg.lstsq(X,target,rcond=None)[0]; loo=[]
  for l in range(4):
   ix=c['lab_id']!=l; loo.extend((X[~ix]@np.linalg.lstsq(X[ix],target[ix],rcond=None)[0]-target[~ix]).tolist())
  print((round(np.std(target-X@fit),4),round(np.sqrt(np.mean(np.square(loo))),4),np.round(fit,3).tolist() if X.shape[1]<8 else ''),end=' ')
 print()
print('amplitudes, gaps, history, drift:');print(np.round(np.c_[A,g,past,hold,f@w[2:]],3))

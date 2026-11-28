exec(open('/tmp/chamber08.py').read().split('ramps=')[0])
keys=('lab_id','stratum','ref_coupons','coupon_features_raw')
Fdiag=np.c_[np.ones(len(H)),_features({k:g[k][:0] for k in keys},g)[1]]
D=Fdiag@th[2:]
def pred(u):
    ramp=2*t/(1+np.sqrt(1+t/u))
    return base+np.outer(D,np.r_[ramp,1.])
prev=runpy.run_path('/holdout_task/batch_solver.py')['_predict_chamber']
c0,r0=prev(tau,th,H,Fdiag,t); old=np.c_[c0,r0]
def metrics(e):
    return np.round([np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])),max(np.median(abs(e[labs==l,:-1])) for l in np.unique(labs)),max(np.median(abs(e[labs==l,-1])) for l in np.unique(labs)),max(np.median(abs(e[:,:-1]),axis=0))],6)
for label,prediction in [('independent weights',old),('coupled unit kernel',pred(1.))]:
    print(label,metrics(prediction-Y),'signed time bias',np.median(prediction-Y,axis=0).round(5))
e=np.zeros_like(Y)
for lab in np.unique(labs):
    s=labs==lab
    u=np.exp(least_squares(lambda v: ((pred(np.exp(v[0]))-Y)[~s]*W).ravel(),[0.],loss='soft_l1',f_scale=.012).x[0]); e[s]=(pred(u)-Y)[s]
    print('held-out lab',lab,'kernel time',round(u,6),'strain/rate',np.round([np.median(abs(e[s,:-1])),np.median(abs(e[s,-1]))],6))
print('fitted coupled OOF',metrics(e)); m=np.isnan(g['coupon_features_raw']).any(axis=1)
for label,s in [('missing',m),('complete',~m)]:
    print(label,s.sum(),np.round([np.median(abs(e[s,:-1])),np.median(abs(e[s,-1]))],6))

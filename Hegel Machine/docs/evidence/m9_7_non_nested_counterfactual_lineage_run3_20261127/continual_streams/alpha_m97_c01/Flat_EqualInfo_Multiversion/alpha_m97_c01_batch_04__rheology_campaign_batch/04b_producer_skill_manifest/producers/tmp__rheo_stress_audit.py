exec(open('/tmp/rheo_cv_final.py').read().split('for lab in (*np.unique(labs), -1):')[0])
out=s.solve_alpha_m97_c01_batch_04(d)
nt=len(d['test']['lab_id'])
for key,shape in {'estimated_taus':(2,), 'hereditary_weights':(7,), 'regime_mode_weights':(2,), 'predicted_strain':(nt,len(t)), 'initial_rate_pred':(nt,)}.items():
    assert out[key].shape==shape and np.isfinite(out[key]).all(),key
u=out['estimated_taus']; h=out['hereditary_weights']; r=out['regime_mode_weights']
z=np.zeros((nt,2))
for j,delta in enumerate(dt):
    e=np.exp(-delta/u)
    z=z*e+(1-e)*past[n:,j,None]
a=np.tanh(np.sum(np.maximum(np.abs(past[n:])-d['crit_stress'],0)**2*dt,axis=1))
fast=h[0]*(hold[n:]-z[:,0])+r[0]*a*hold[n:]
slow=h[1]*(hold[n:]-z[:,1]); drift=h[2]+x[n:]@h[3:]+r[1]*a*z[:,0]
base=np.asarray(d['test']['strain_0'])
def curve(t):
    t=np.asarray(t)
    return base[:,None]-fast[:,None]*np.expm1(-t/u[0])-slow[:,None]*np.expm1(-t/u[1])+drift[:,None]*np.log1p(t)
np.testing.assert_allclose(out['predicted_strain'],curve(t),atol=1e-12,rtol=0)
np.testing.assert_allclose(out['initial_rate_pred'],fast/u[0]+slow/u[1]+drift,atol=1e-12,rtol=0)
fd=(curve([1e-5])[:,0]-curve([0.0])[:,0])/1e-5
for label,mask in [('undamaged',a==0),('damaged',a>0)]:
    delta=np.max(np.abs(fd[mask]-out['initial_rate_pred'][mask]))
    print(label,'test rows',mask.sum(),'max rate/finite-difference error',delta); assert delta<1e-4
# Fit omitted drift terms inside each outer fold. A transferable improvement
# on complete coupons would warrant revisiting the stated linear drift.
v=x[:n]
pairs=[(v[:,i],v[:,j]) for i in range(4) for j in range(i+1,4)]
extra=np.column_stack([term for p,q in pairs for term in (p*q,(p+q)/(1+(p-q)**2),(p-q)/(1+(p+q)**2))]+[v[:,i]*v[:,j]*v[:,k] for i,j,k in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]])
loss=[]
for lab in np.unique(labs):
    f=(labs!=lab)&~missing; g=(labs==lab)&~missing
    tau,w=s._fit(dt,t,*(a[f] for a in args),yc[f],yr[f],~missing[f])
    c,b=s._design(tau,dt,t,*args)
    ec=yc-c@w; er=yr-b@w
    row=np.median(ec/np.log1p(t),axis=1)
    adjustment=extra@np.linalg.solve(extra[f].T@extra[f]+np.eye(extra.shape[1]),extra[f].T@np.clip(row[f],-.07,.07))
    metrics=[np.median(np.abs(ec[g])),np.median(np.abs(ec[g]-adjustment[g,None]*np.log1p(t))),np.median(np.abs(er[g])),np.median(np.abs(er[g]-adjustment[g]))]
    loss.append(metrics); print('lab',lab,'curve baseline/augmented, rate baseline/augmented',np.round(metrics,6))
print('worst lab',np.max(loss,axis=0).round(6))

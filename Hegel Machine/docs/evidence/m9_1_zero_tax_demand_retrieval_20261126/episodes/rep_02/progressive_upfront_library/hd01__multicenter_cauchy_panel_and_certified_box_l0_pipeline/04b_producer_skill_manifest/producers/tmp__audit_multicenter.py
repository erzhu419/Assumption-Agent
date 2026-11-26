import io
import contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from verify_multicenter import solver, check, np, public

for i, entry in enumerate(public):
    cohort=entry['study']['cohort']; gt=entry['gt']; c=cohort['center_id']; t=cohort['stratum']; y=cohort['y_potency']; z=solver._align(cohort); beta=gt['beta_potency']; seen=np.isfinite(cohort['X_raw'])
    slopes=[]; adjusted=[]; cells=[]
    for k in np.unique(c):
        tr=c!=k; centers=np.unique(c[tr]); indicators=(c[tr,None]==centers[None,1:]).astype(float)
        a=solver._potency(z[tr],t[tr],y[tr]); trial=solver._potency(np.column_stack((z[tr],indicators)),t[tr],y[tr])[:z.shape[1]]
        slopes.append(np.linalg.norm(a-beta)/np.linalg.norm(beta)); adjusted.append(np.linalg.norm(trial-beta)/np.linalg.norm(beta))
        for s in np.unique(t):
            r=(c==k)&(t==s); e=np.abs(z[r]-gt['Z_true'][r]); cells.append((e.mean(),e[seen[r]].mean(),int(k),int(s)))
    print('study',i,'LOCO slope baseline/extra center intercept',np.round(slopes,5),np.round(adjusted,5),'worst aligned cell',max(cells),flush=True)

# Clustered scores close to activation are where a root-wide diagonal floor
# can force many unnecessary branches even when the Gram is well-conditioned.
rng=np.random.default_rng(398721); worst=[]
for i in range(20):
    p=22; l0=.22; l2=.14; m=1.1; g=np.eye(p); j=np.arange(p-1)
    g[j,j+1]=g[j+1,j]=rng.choice([-1,1],p-1)*rng.uniform(.16,.32,p-1)
    x=np.linalg.cholesky(g).T
    b=rng.choice([-1,1],p)*np.sqrt(2*l0*(1+l2))*rng.uniform(.98,1.02,p)
    y=np.linalg.solve(x.T,b); s=dict(X=x,y=y,M=m,lam0=l0,lam2=l2)
    try:
        leaves,gap=check(s,solver._certify(s)); worst.append((leaves,gap,i))
    except RuntimeError as e:
        print('near activation failure',i,e,flush=True); np.savez('/tmp/multicenter_activation_%02d.npz'%i,**s)
print('near activation certified',len(worst),'largest proofs',sorted(worst,reverse=True)[:5],flush=True)

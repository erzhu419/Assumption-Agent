exec(open('/tmp/probe14.py').read().split("u=s['panel_a']")[0])
u,v=[s['panel_a'][k] for k in ('u','v')]; a=design(u,terms[:15]); _,r=fit(u,v,terms[:15],True); good=r<5*np.median(r);print('outlier norm',np.sort(r).round(3));
for d in (1,2,3):
 ts=[t for t in terms if len(t)<=d];a=design(u,ts);c=np.linalg.lstsq(a[good],v[good],rcond=None)[0];print('A degree',d,'rmse',np.sqrt(np.mean((a[good]@c-v[good])**2)));print('terms',[(t,c0.round(3).tolist()) for t,c0 in zip(ts,c) if max(abs(c0))>.025]);
F=terms[:5]+[(j,j) for j in range(4)]; u=s['panel_c']['u'];v=s['panel_c']['v2']; c,r=fit(u,v,F);print('C2 diagonal rat',np.round(c[5:]/c[1:5],3),'rms',np.sqrt(np.mean(r*r))); print('state singular',np.linalg.svd(s['train']['traj'],compute_uv=False))

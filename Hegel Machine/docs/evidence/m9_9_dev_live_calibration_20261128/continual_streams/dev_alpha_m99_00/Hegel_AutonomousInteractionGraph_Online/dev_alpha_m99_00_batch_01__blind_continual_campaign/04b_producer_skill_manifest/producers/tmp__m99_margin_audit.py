exec(open('/tmp/m99_joint.py').read().split('models={')[0])
D = np.c_[one, m._features(di)] @ th[2:]
complete = np.isfinite(di['coupon_features_raw']).all(axis=1)
all_rows = np.arange(N)
slices = {f'lab{lab[g[0]]}': g for g in groups}
slices.update(imputed=all_rows[~complete], complete=all_rows[complete], low_hold=np.argsort(h)[:6], high_hold=np.argsort(h)[-6:])
H, J = h / np.median(h), D / np.std(D)
models = {
 'affine': F(one, h, D),
 'through_origin': F(h,D),
 'pairwise_product': F(one,h,D,(H+J)*(H-J),H*J),
 'bounded_quotient': F(one,h,D,H/(1+J**2),J/(1+H**2)),
}
for name, A in models.items():
 rr=np.empty_like(Y)
 for g in groups:
    rr[g]=fit((A,)*3,np.setdiff1d(all_rows,g))[g]
 print(name, 'pooled', fmt(rr), 'early/late', np.round([np.sqrt(np.mean(rr[:,:3]**2)), np.sqrt(np.mean(rr[:,-4:-1]**2))],5))
 print({key:fmt(rr[ix]).tolist() for key, ix in slices.items()})
for key, ix in slices.items():
 if key.startswith('lab'):
    print(key, 'train/test labs represented', int(np.sum(tr['lab_id']==lab[ix[0]])), int(np.sum(te['lab_id']==lab[ix[0]])))

exec(open('/tmp/check11.py').read().split('print(\'cross coeff\'')[0])
for label, te in [('required',np.arange(52,72)), *[(f'fold{i}',v) for i,v in enumerate(np.array_split(np.random.default_rng(11).permutation(72),4))], *[(f'tail{i}',np.argsort(abs(z[:,i]))[-20:]) for i in range(4)], ('high-radius',np.argsort(np.linalg.norm(z,axis=1))[-20:])]:
 ix = np.setdiff1d(np.arange(len(x)),te)
 basis=m._time_basis(s['eval_times'],T[ix]); target=T[ix]@basis
 cf=m._fit(f[ix],target)[-c.shape[1]:]
 support=np.argmax(np.linalg.norm(cf,axis=1)); g=np.column_stack((b,c[:,support]))
 def trajectory(d):
  e=abs((d[te]@m._fit(d[ix],target))@basis.T-T[te]); return tuple(np.round([e.mean(),e.mean(0).max(),e[:,-1].mean()],6))
 print(label,ij[support],'sparse',trajectory(g),'dense',trajectory(f))

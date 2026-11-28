exec(open('/tmp/stress04.py').read().split("print('scalar full coeff'")[0])
rng=np.random.default_rng(677)
splits=[np.arange(72)]+[rng.permutation(72) for _ in range(30)]+[np.argsort(np.linalg.norm(A[:72,1:],axis=1))]
for mix in (0.,.25,.5,.75,1.):
 rows=[]; test=[]
 for split in splits:
  i,j=split[:52],split[52:]; pair=D[:,[0,1,2,3,4,7,11]]
  b=m._ls(pair[i],y[i]); r=m._ls(D[i],y[i]); e=np.abs((1-mix)*(pair[j]@b)+mix*(D[j]@r)-y[j]);rows.append([e.mean(),e.max()]);test.append((1-mix)*(pair[72:]@b)+mix*(D[72:]@r))
 print(mix,'primary',rows[0],'random worst',np.max(rows[:-1],0),'tail',rows[-1],'max test split variability',np.std(test,axis=0).max())

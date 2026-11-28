exec(open('/tmp/readout_probe.py').read().split("print('top single residual features')")[0])
from scipy.optimize import least_squares
base_c=np.c_[np.ones(n),c]; pair=c[:,1]*c[:,3]; f=np.c_[base_c,pair]
print('interaction',score(f),base(f,y))
res=y-f@base(f,y)
print('strongest second terms')
print(*sorted((score(np.c_[f,v]),k,base(np.c_[f,v],y)[-1]) for k,v in zip(names,cols) if np.linalg.norm(v-f@base(f,v))>1e-8)[:16],sep='\n')
q=np.column_stack([base_c]+[c[:,i]*c[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)])
print('full quadratic',score(q),base(q,y))
for name,extra in [('v2',b),('z',z),('both',np.c_[z,b]),('traj modes',np.linalg.svd(tr,full_matrices=False)[0][:,:2])]:
 h=np.c_[f,extra]; print(name,score(h),base(h,y))
sets=[]
for freq in (.25,.5,1.,1.5,2.,3.):
 for i,j in itertools.combinations(range(4),2):
  a,d=c[:,i],c[:,j]
  for tag,w in [('sin+',np.sin(freq*(a+d))),('sin-',np.sin(freq*(a-d))),('cos+',np.cos(freq*(a+d))),('cos-',np.cos(freq*(a-d))),('tanhprod',np.tanh(freq*a*d)),('sinprod',np.sin(freq*a*d)),('ratprod',a*d/(1+freq*(a*d)**2))]:
   sets.append((f'{tag}_{i}{j}_{freq}',np.c_[base_c,w]))
print('replace product',*sorted((score(h),key) for key,h in sets)[:10],sep='\n')
models={'affine':base_c,'product':f,'full quadratic':q,'product + z':np.c_[f,z],'product + v2':np.c_[f,b]}
for name,h in models.items():
 scores=[]; corners=[]
 for inds in np.array_split(np.random.default_rng(812).permutation(n),6)+[np.argsort(np.linalg.norm(z,axis=1))[-12:]]:
  keep=np.setdiff1d(np.arange(n),inds); e=abs(y[inds]-h[inds]@base(h[keep],y[keep])); scores.append(e.mean()); corners.append(e.max())
 print('stress',name,np.round(scores,5),'worst row',max(corners))

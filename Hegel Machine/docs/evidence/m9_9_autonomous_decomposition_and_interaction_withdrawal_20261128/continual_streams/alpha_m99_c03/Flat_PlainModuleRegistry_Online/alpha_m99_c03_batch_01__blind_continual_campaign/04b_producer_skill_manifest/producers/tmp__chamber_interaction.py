exec(open('/tmp/chamber_memory.py').read().split("print('zero-drift")[0])
print('zero-drift q, RMS each output',q,np.sqrt(np.mean(prof(q).reshape(N,-1)**2,axis=0)))
print('a minus intrinsic hold RMS and correlation',np.std(a-th[:2]*h[:,None],0),np.corrcoef((a-th[:2]*h[:,None]).T)[0,1])
print('drift index and per-coupon stress corrections\n',np.c_[d@th[2:],a/th[:2]-h[:,None]].round(4))
from itertools import combinations
from scipy.stats import spearmanr
after=np.cumsum(dt[::-1])[::-1]-dt
mem=lambda s:p@(-np.expm1(-dt[:,None]/s)*np.exp(-after[:,None]/s))
z=mem(q); v=a/th[:2]-h[:,None]; v1=v[:,1]; C={}
def add(names,x): C.update(zip(names,np.asarray(x).reshape(N,-1).T))
add(['hold','zfast','zslow','strain0','lab','stratum','drift'],np.c_[h,z,dg['strain_0'],lab,dg['stratum'],d@th[2:]])
for prefix,mat in [('x',d[:,1:]),('x_impute_raw',np.nan_to_num(dg['coupon_features_raw'])),('ref',dg['ref_coupons']),('p',p),('pc',dg['past_channel_raw'].reshape(N,-1)),('hc',dg['hold_channel_raw'])]: add([prefix+str(j) for j in range(mat.shape[1])],mat)
keys=['hold','zfast','zslow','strain0','drift']+[f'x{k}' for k in range(4)]+[f'p{k}' for k in range(6)]
for i,j in combinations(keys,2):
 x,w=C[i],C[j]; C[i+'*'+j]=x*w; C[i+'-'+j]=x-w; C[i+'+'+j]=x+w
for i in keys:
 x=C[i]; C[i+'^2']=x*x; C['abs('+i+')']=abs(x); C['pos('+i+')']=np.maximum(x,0)
for k in range(4): C[f'lab{k}']=(lab==k).astype(float)
for k in range(3): C[f'stratum{k}']=(dg['stratum']==k).astype(float)
print('top Pearson',sorted([(k,*np.corrcoef(np.c_[x,v].T)[0,1:].round(3)) for k,x in C.items()],key=lambda w:-abs(w[-1]))[:25]); print('top rank',sorted([(k,round(spearmanr(x,v1).statistic,3)) for k,x in C.items()],key=lambda w:-abs(w[-1]))[:15]);
for l in np.unique(lab): print('lab',l,'mean/std',v[lab==l].mean(0),v[lab==l].std(0))
for label,F in [('hold',np.c_[h]),('drift',np.c_[h,d]),('stress',np.c_[h,z,p,np.ones(N)]),('all main',np.c_[h,z,p,d,dg['strain_0']]),('ref',np.c_[h,dg['ref_coupons'],np.ones(N)])]:
 E=[]
 for l in np.unique(lab):
  msk=lab!=l; coef=np.linalg.lstsq(F[msk],v[msk],rcond=None)[0]; E.append(np.sqrt(np.mean((F[~msk]@coef-v[~msk])**2,0)))
 print(label,'worst branch stress RMS',np.max(E,0).round(5),'coefs',np.linalg.lstsq(F,v,rcond=None)[0].round(4).T)

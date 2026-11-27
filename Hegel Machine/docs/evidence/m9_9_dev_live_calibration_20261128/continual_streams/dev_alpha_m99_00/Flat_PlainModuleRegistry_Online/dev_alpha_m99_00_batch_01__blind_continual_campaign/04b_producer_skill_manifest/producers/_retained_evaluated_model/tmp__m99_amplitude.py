import contextlib,io,itertools
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/m99_probe.py').read())
p,h,D=cc; labs=ch['lab_id']; Y=np.c_[ch['strain_curves_obs']-ch['strain_0'][:,None],ch['instant_rate_obs']]; H=np.r_[-np.expm1(-t[:,None]/tau)@theta[:2],theta[:2]@(1/tau)]; K=np.r_[np.log1p(t),1.]; a=(Y-h[:,None]*H)@K/(K@K)
X=D[:,1:]; r=D@theta[2:]; rawf=ch['coupon_features_raw']; flags=np.isnan(rawf); base=np.c_[np.ones(len(h)),r]; cells=ch['stratum']
print('stratum residuals',[(j,(a-r)[cells==j].mean(),(a-r)[cells==j].std()) for j in np.unique(cells)])
Q={'r':r,'h':h,'hcenter':h-h.mean(),'pastmean':p.mean(1),'pastlast':p[:,-1]}; Q.update({'x%d'%j:X[:,j] for j in range(4)}); Q.update({'raw%d'%j:np.nan_to_num(rawf[:,j]) for j in range(4)}); Q.update({'p%d'%j:p[:,j] for j in range(p.shape[1])}); Q.update({'flag%d'%j:flags[:,j].astype(float) for j in range(4)}); Q.update({'lab%d'%j:(labs==j).astype(float) for j in np.unique(labs)}); Q.update({'str%d'%j:(cells==j).astype(float) for j in np.unique(cells)}); orig=list(Q)
for l,k in itertools.combinations_with_replacement(orig[:13],2): Q[l+'*'+k]=Q[l]*Q[k]
for l,k in itertools.combinations(orig[:13],2):
 for s in [1,-1]:
  v=Q[l]+s*Q[k]; name=l+('+' if s==1 else '-')+k
  for fun,tag in [(np.abs,'abs'),(np.square,'sq'),(np.tanh,'tanh')]: Q[tag+'('+name+')']=fun(v)
for l,k,j in itertools.combinations_with_replacement(orig[5:9],3): Q[l+'*'+k+'*'+j]=Q[l]*Q[k]*Q[j]
for l,k in itertools.product(orig[:9],orig[:9]): Q[l+'/(1+'+k+'^2)']=Q[l]/(1+Q[k]**2)
for bname,B in [('material',base),('features',D),('per-lab',np.c_[base,np.eye(4)[labs][:,:3]]),('per-stratum',np.c_[base,np.eye(3)[cells][:,:2]])]:
 def score(M):
  E=np.zeros(len(h));
  for lab in np.unique(labs):
   q=labs!=lab; E[~q]=a[~q]-M[~q]@np.linalg.lstsq(M[q],a[q],rcond=None)[0]
  return np.sqrt(np.mean(E**2)),abs(E).max()
 print(bname,'base',score(B))
 out=[]
 for n,v in Q.items():
  M=np.c_[B,v]; coef=np.linalg.lstsq(M,a,rcond=None)[0]; E=a-M@coef; out.append((score(M)[0],np.std(E),n,coef[-1]))
 print('by LOO',sorted(out)[:12]); print('by fit',sorted(out,key=lambda z:z[1])[:8])

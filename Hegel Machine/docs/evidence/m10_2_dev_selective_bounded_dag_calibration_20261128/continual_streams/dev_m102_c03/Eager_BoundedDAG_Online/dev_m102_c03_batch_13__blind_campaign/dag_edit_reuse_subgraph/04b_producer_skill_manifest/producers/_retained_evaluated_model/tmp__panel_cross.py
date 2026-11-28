exec(open('/tmp/panel_probe.py').read().split("for i,j in [(0,0)")[0])
roll=lambda a,k=1:np.roll(a,k,axis=1)
for name in ['v1','v2']:
 v=p[name]; ans=[]
 for label,g in lib.items():
  for shift in [-1,1]:
   for inter in ['none','prod','linear','cross','sum','diff']:
    def design(a,k):
     if inter=='sum': return bias(g(a+k[1]*roll(a,shift),k[0]))
     if inter=='diff': return bias(np.c_[a,g(a+k[1]*roll(a,shift),k[0])])
     m=g(a,k[0]); q=roll(a,shift)
     extra={'none':[], 'prod':[a*q], 'linear':[a], 'cross':[a,a*q]}[inter]
     return bias(np.column_stack([m,*extra]))
    def obj(k):
     f=design(u,k); return (f@ls(f,v)-v).ravel()
    if label=='linear' and inter in ['sum','diff']: continue
    for guess in [[.85,.2],[2.15,-.2]]:
     if inter not in ['sum','diff']: guess=guess[:1]
     r=least_squares(obj,guess,bounds=([.01,-1][:len(guess)],[5.,1][:len(guess)]),max_nfev=60)
     ans.append((np.std(r.fun),label,shift,inter,r.x))
 print(name,sorted(ans,key=lambda x:x[0])[:14])

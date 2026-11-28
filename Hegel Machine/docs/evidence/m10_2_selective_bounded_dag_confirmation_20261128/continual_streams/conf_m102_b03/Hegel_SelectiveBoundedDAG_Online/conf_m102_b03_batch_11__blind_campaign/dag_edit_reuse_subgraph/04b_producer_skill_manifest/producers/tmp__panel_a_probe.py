import pickle,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v')); rng=np.random.default_rng(13)
for j in range(4):
 print('\nOUT',j)
 za=[u[:,(j+i)%4] for i in range(4)]; z,w,t,h=za
 def matrix(fs):return np.column_stack([np.ones(len(z))]+fs)
 cases={'diag':[z], 'full':za,'arch':[z,w,z*w,z**3], 'prev':[z,h,z*h,z**3],'opp':[z,t,z*t,z**3], 'full+own':[ *za,z*z,z**3], 'all2':za+[a*b for a,b in itertools.combinations_with_replacement(za,2)],'all2+cubes':za+[a*b for a,b in itertools.combinations_with_replacement(za,2)]+[a**3 for a in za], 'archsat':[z,w,z*w,z/(1+z*z)],'sat':[ *za,z/(1+z*z),z*z/(1+z*z)], 'archtanh':[z,w,z*w,np.tanh(z)]}
 for k,fs in cases.items():
  a=matrix(fs); d=a.shape[1]; y=v[:,j]; best=np.inf
  for _ in range(450):
   g=rng.choice(len(y),d+1,replace=False); b=np.linalg.lstsq(a[g],y[g],rcond=None)[0]; e=abs(a@b-y); val=np.quantile(e,.58)
   if val<best:best=val; bb=b
  for _ in range(4):
   e=abs(a@bb-y); g=e<=max(.001,4*np.median(e));bb=np.linalg.lstsq(a[g],y[g],rcond=None)[0]
  print(k, np.round(np.sort(np.abs(a@bb-y))[[20,28,34,40,43]],5), np.round(bb,3),np.where(~g)[0] if k=='all2+cubes' else '')
print('A pairs'); print(np.round(np.column_stack([u,v]),3))

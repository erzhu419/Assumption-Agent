exec(open('/tmp/modelprobe.py').read().split('o,M=')[0]); from itertools import combinations
u,v=A['u'],A['v']; ii,jj=np.triu_indices(4); mask=np.ones(len(u),bool)
def fit(h):
 mask=np.ones(len(u),bool)
 for _ in range(12):
  w=ls(h[mask],v[mask]); e=np.linalg.norm(h@w-v,axis=1); new=e<max(4*np.median(e),.003)
  if np.array_equal(mask,new):break
  mask=new
 return w,e,mask
h=np.c_[a(u),u[:,ii]*u[:,jj]]; w,e,mask=fit(h); print('linear and quadratic pairs',list(zip(ii,jj)),w,sep='\n'); print('quant',np.quantile(e[mask],[0,.5,.9,1]));
for j in range(4):
 print('output',j)
 for i in range(4):
  if i==j:continue
  for typ in range(5):
   def hfun(k):
    z=u[:,i]; b=u[:,j]; f=[b*z/(1+k*z*z),z/(1+k*z*z),z*z/(1+k*z*z),b*z/(1+k*b*b),np.tanh(k*z)][typ]; return np.c_[a(u),f][mask]
   def err(k):
    g=hfun(k[0]);return g@ls(g,v[mask,j])-v[mask,j]
   k=least_squares(err,[.4],bounds=(.001,10)).x[0]; g=hfun(k); rm=np.sqrt(np.mean(err([k])**2));
   if rm<.065:print(i,typ,k,rm,ls(g,v[mask,j]))

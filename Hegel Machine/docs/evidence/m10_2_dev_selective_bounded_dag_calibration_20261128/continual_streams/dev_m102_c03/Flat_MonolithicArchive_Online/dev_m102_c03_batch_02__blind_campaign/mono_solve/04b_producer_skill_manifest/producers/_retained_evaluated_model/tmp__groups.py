exec(open('/tmp/readout.py').read().split('base=np.c_')[0]); from itertools import product
print('cal',g,o,'B',b,'Berr',rms(B['v']-aff(B['u'])@b)); print('C2',np.linalg.lstsq(aff(np.cos(2.4*p)),C['v2'],rcond=None)[0]); print('rate',np.exp(r.x))
z0=z; S=z/(1+.83*abs(z)); G={'z':z,'D':D,'V':V,'U':U,'B':BZ,'S':S}; one=np.ones(len(z));
for j,k in product(G,G):
 if j>k:continue
 a,c=G[j],G[k]; G0=(a[:,:,None]*c[:,None,:]).reshape(len(z),16); 
 fs=[('diag',a*c),('full',G0)];
 for ty,F in fs:
  for bas,base0 in [('zero',one[:,None]),('zD',np.c_[one,z,D]),('zDU',np.c_[one,z,D,U])]:
   H=np.c_[base0,F]; w=np.linalg.lstsq(H[:n],Y[:n],rcond=1e-7)[0]; e=H@w-Y; ee=np.r_[mae(e[n:]),mae((e[n:,1:]@q.T).ravel())];
   if min(ee)<.05: print(j,k,ty,bas,'rms',rms(e[:n]),'val',ee)
M=U[:,:,None]*V[:,None,:]; F=np.c_[one,z,D,U,M.reshape(72,16)]; w=np.linalg.lstsq(F[:n],Y[:n],rcond=1e-7)[0]; print('UV intercept and main effects',w[:13]); print('UV interaction',w[13:].reshape(4,4,3).transpose(2,0,1));
for i in range(4):
 for j in range(4):
  ff=np.c_[one,M[:,i,j]]; w=np.linalg.lstsq(ff[:n],Y[:n],rcond=None)[0]; e=Y-ff@w
  if min(rms(e[:n]))<.04:print('single',i,j,w, 'rms',rms(e[:n]),'val',mae(e[n:]));
# Estimate temporal response from a measured candidate slow amplitude.
h=U[:,0]*V[:,1]; H=np.c_[one,h]; W=np.linalg.lstsq(H[:n],T[:n],rcond=None)[0]; print('h response regression',W)

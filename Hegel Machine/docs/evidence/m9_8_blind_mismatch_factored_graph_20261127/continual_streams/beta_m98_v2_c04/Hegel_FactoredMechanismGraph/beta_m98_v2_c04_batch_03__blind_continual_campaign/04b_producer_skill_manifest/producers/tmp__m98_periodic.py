import itertools
from m98_walls import *
def candidate(d, key):
 inward, wrap, adv=key
 k,v,L=s['kappa_'+d],s['vel_'+d],s['L'+d]; h=L/(len(k)-1)
 a=mod._axis(k,v,L,0.).toarray()
 for i,j,t,sgn in ((0,1,-1,-1),(-1,-2,0,1)):
  ki=k[i] if inward=='local' else (k[i]+k[j])/2
  kw={'mean':(k[i]+k[t])/2,'local':k[i],'remote':k[t],'harmonic':2*k[i]*k[t]/(k[i]+k[t])}[wrap]
  vi,vw={'upwind':(max(-sgn*v[i],0),max(sgn*v[i],0)), 'none':(0,0),'swap':(max(sgn*v[i],0),max(-sgn*v[i],0)), 'centered':(-sgn*v[i]/2,sgn*v[i]/2)}[adv]
  a[i,j]=-ki/h**2-vi/h; a[i,t]=-kw/h**2-vw/h; a[i,i]=-a[i,j]-a[i,t]
 return csc_matrix(a)
keys=list(itertools.product(('local','mean'),('mean','local','remote','harmonic'),('upwind','none','swap','centered')))
A,B=([{k:candidate(d,k) for k in keys} for d in 'xy']); results=[[],[],[]]
for a,b in itertools.product(keys,repeat=2):
 R=predict(A[a],B[b])-Y; E=scores(profile(R)); rms=np.sqrt(np.mean(E**2,axis=0))
 for r,val in zip(results,(*rms,np.linalg.norm(rms))):r.append((val,a,b,E,np.sqrt(np.mean(R**2))))
for title,r in zip(('pulse0','pulse1','joint'),results):
 print(title)
 for val,a,b,E,raw in sorted(r,key=lambda x:x[0])[:4]: print(a,b,'RMSE',np.round(E,8).tolist(),'raw joint',raw)

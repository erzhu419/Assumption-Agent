import numpy as np
import weights,temporal as m
n=52;s=m.s; raw=m.raw[:72];x=m.x[:72];c=m.c[:72];d=m.d[:72];Y=np.c_[m.y,m.T];
for desc,tar in [('joint',Y),('scalar',Y[:,:1]),('trajectory',np.c_[Y[:,1],Y[:,1:]])]:
 p=weights._joint(s,raw,x,c,d,tar[:n]); print(desc,np.mean(abs(p-tar[n:]),0))
print('t',m.tt)

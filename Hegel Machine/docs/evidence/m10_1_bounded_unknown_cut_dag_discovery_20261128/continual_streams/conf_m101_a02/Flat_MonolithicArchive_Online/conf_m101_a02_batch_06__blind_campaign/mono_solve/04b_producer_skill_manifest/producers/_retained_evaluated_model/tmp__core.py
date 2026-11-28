from operators import *
from itertools import product
b,c,d=[ops[k](z) for k in 'BCD']; n=72
Z=aff(z); A=np.c_[Z,b,c,d]; I=np.arange(n); q=(b*b)@np.array([1,1,-1,-1]); print('qscale',q[:n].std(),np.corrcoef(q[:n],Y[:,0])[0,1]);
for nm,base in [('z',Z),('b',aff(b)),('zb',np.c_[Z,b]),('zbc',np.c_[Z,b,c]),('zbd',np.c_[Z,b,d]),('all',A)]:
 for fixed in (False,True):
  F=base if fixed else np.c_[base,q]; yy=Y[:,0]-fixed*q[:n]; ss=[]
  for i,j in folds: ss.append(np.mean(abs(F[j]@fit(F[i],yy[i])-yy[j])))
  print('CORE',nm,fixed,ss,'rstd',np.std(F[:n]@fit(F[:n],yy)-yy))
# Fit a signed quadratic spanning B and a small correction from C/D/z, preserving its signature.
for hname,h in [('z',z),('c',c),('d',d),('zc',np.c_[z,c]),('zcd',np.c_[z,c,d])]:
 for mode in ('common','diag','matrix'):
  for reg in (.03,.15,.6,2.):
   es=[]
   for i,j in folds:
    a=np.c_[Z,b,h]; P=np.linalg.qr(a[i])[0]; yy=Y[i,0]; sg=np.array([1,1,-1,-1]);
    if mode!='matrix' and h.shape[1]!=4:continue
    def qf(p):
     g=b+(h*p if mode in ('common','diag') else h@p.reshape(-1,4));return g*g@sg
    def res(p):
     r=qf(p)[i]-yy;return np.r_[r-P@(P.T@r),reg*p]
    p=least_squares(res,np.zeros(1 if mode=='common' else 4 if mode=='diag' else 4*h.shape[1])).x
    qq=qf(p);es.append(np.mean(abs(qq[j]+a[j]@fit(a[i],yy-qq[i])-Y[j,0])))
   if es:print('MIX',hname,mode,reg,np.round(es,5))

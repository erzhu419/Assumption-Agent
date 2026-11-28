exec(open('/tmp/probe.py').read().split('res=[]')[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
u=s['panel_c']['u']; v1=s['panel_c']['v1']; v2=s['panel_c']['v2']; A=affine(u)
print('panel B singular',np.linalg.svd(matrices['panel_bv'][1:],compute_uv=False));print('v1 range',v1.min(0),v1.max(0),'v2 range',v2.min(0),v2.max(0))
for j,Y in enumerate(v1.T):
 print('C1 channel',j)
 for name,fun in [('tanh',np.tanh),('sin',np.sin),('rational',lambda v:v/np.sqrt(1+v*v)),('atan',np.arctan),('linear',lambda v:v)]:
  def pred(p):return p[0]*fun(A@p[1:6])+p[6]
  p=least_squares(lambda p:pred(p)-Y,np.r_[max(1.,max(abs(Y))),np.linalg.lstsq(A,Y,rcond=None)[0],0],max_nfev=1000).x
  print(name,np.sqrt(np.mean((pred(p)-Y)**2)),p)
Phi=basis(u,2); W=np.linalg.lstsq(Phi,v2,rcond=None)[0];print('v2 quad coefs',W.T,'res',np.sqrt(np.mean((Phi@W-v2)**2,axis=0)))
for name,U in [('x',x),('ca',Q['caldiag']),('B',Q['xpanel_bv']),('C',Q['xpanel_cv1']),('V2',Q['xpanel_cv2'])]:
 D=affine(U);w=np.linalg.lstsq(D,y,rcond=None)[0];r=y-D@w; print('y linear',name,w,'mae',np.mean(abs(r)),'max',max(abs(r)))
 for d in [2,3]:
  D=basis(U,d); w=np.linalg.lstsq(D,y,rcond=None)[0];print(' y poly',d,np.mean(abs(y-D@w)),sorted(enumerate(w),key=lambda p:-abs(p[1]))[:9])
print('largest residual raw',np.c_[x,y,(y-affine(x)@np.linalg.lstsq(affine(x),y,rcond=None)[0])][np.argsort(abs(y-affine(x)@np.linalg.lstsq(affine(x),y,rcond=None)[0]))[-12:]])

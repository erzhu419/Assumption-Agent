from model import *
tr=s['train'];y=tr['y'];t=tr['traj'];times=s['eval_times']
def prof(A,z):return z-A@ls(A,z)
for j in range(4):
 a=u[:,j];b=u[:,(j+2)%4];tar=v[:,j];out=[]
 for n,f in funcs().items():
  for where in ['self','cross','sum','diff','both']:
   def D(p):
    hs={'self':[f(a,p[0])],'cross':[f(b,p[0])],'sum':[f(a+b,p[0])],'diff':[f(a-b,p[0])],'both':[f(a,p[0]),f(b,p[0])]}
    return np.c_[np.ones(len(a)),a,b,*hs[where]]
   k=min([least_squares(lambda p:prof(D(p),tar),[q],bounds=(.04,4)) for q in [.3,1,2.3]],key=lambda x:x.cost);out.append((rm(k.fun),n,where,k.x,ls(D(k.x),tar)))
 print('cal j',j,sorted(out,key=lambda x:x[0])[:5])
F=lambda p:-np.expm1(-times[:,None]*p)
for p0 in [[.3,1.7],[.22,2.15],[1.25,2.15]]:
 p=least_squares(lambda p:prof(F(p),t[:52].T).ravel(),p0,bounds=(.0001,8)).x
 print('rates',p,rm(prof(F(p),t.T)));print('telemetry rates',rm(prof(F([1.25,2.15]),t.T)),rm(prof(F([.22,1.25]),t.T)))
cal=fitcal(funcs()['tanh'],2); x=calibrate(tr['x'],funcs()['tanh'],cal[1],cal[2],2)
b=s['panel_b']; B=ls(af(b['u']),b['v']);
def Cbasis(x,k):return np.c_[af(x),funcs()['soft'](x,k)]
C=ls(Cbasis(c,.83),v1);Q=ls(np.c_[af(c),c*c],v2)
np.set_printoptions(precision=4,suppress=True);print('cal',cal);print('C1',C,'C2',Q)
Z=np.c_[y,ls(F(p),t.T).T]; Z=Z/Z[:52].std(axis=0)
for name,h in [('x',x),('B',af(x)@B),('C',Cbasis(x,.83)@C),('Q',np.c_[af(x),x*x]@Q),('C(B)',Cbasis(af(x)@B,.83)@C),('Q(B)',np.c_[af(af(x)@B),(af(x)@B)**2]@Q),('B(C)',af(Cbasis(x,.83)@C)@B),('B(Q)',af(np.c_[af(x),x*x]@Q)@B)]:
 print(name,'range',abs(h).max());models={'aff':af(h),'quad':quad(h),'addsoft':np.c_[af(h),funcs()['soft'](h,.83)],'addtanh':np.c_[af(h),np.tanh(h)],'quadr+soft':np.c_[quad(h),funcs()['soft'](h,.83)]}
 for n,d in models.items():
  w=ls(d[:52],Z[:52]);out=d@w;err=out-Z
  print(n,np.round(mae(err[:52]),4),np.round(mae(err[52:]),4),round(mae((err[52:,1:]*ls(F(p),t.T).T[:52].std(axis=0))@F(p).T).mean(),4))

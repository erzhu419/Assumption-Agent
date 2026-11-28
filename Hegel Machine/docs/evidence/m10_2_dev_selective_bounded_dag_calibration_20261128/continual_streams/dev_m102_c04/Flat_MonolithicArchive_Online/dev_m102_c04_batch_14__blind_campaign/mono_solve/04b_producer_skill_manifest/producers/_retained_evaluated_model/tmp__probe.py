from identify import *
import importlib.util
sp=importlib.util.spec_from_file_location('m','/tmp/hegel_continual_archive/dev_m102_c04_batch_11_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
pa=s['panel_a']; u,v=pa['u'],pa['v']; keep=np.max(abs(u-v),axis=1)<3; print('cal rows',keep.sum()); print('cal inverse discrepancy',np.max(abs(m._calibrate(pa,v[keep])-u[keep]),axis=0));
x=s['train']['x']; zz=[x,m._calibrate(pa,x),(x-np.array([ls(D(u[keep,j]),v[keep,j])[0] for j in range(4)]))/np.array([ls(D(u[keep,j]),v[keep,j])[1] for j in range(4)])]; C=s['panel_c']; u=C['u']; v=C['v1']; r=C['v2'];
def h(x,k):return x/np.sqrt(1+k*x*x)
rf=lambda z,k:D(h(z,k),off(z)[:,[0,2,3,5]])
f=minimize_scalar(lambda k:np.linalg.norm((a:=rf(u,k))@ls(a,r)-r),bounds=(.2,1.5),method='bounded'); print('v2 sparse',f.x,f.fun/np.sqrt(r.size), ls(rf(u,f.x),r));
# Profile shared shape and inspect linear/nonlinear proportions.
for name in ('tanh','rat','soft','powrat','atan'):
 fn=funcs[name]; ff=lambda z,k:D(z,fn(z,k)); f1=minimize_scalar(lambda k:np.linalg.norm((a:=ff(u,k))@ls(a,v)-v),bounds=(.02,2.8),method='bounded'); co=ls(ff(u,f1.x),v); print(name,f1.x,f1.fun/np.sqrt(v.size), np.round(co[1:5]/co[5:],2));
# Search componentwise nonlinearities, including linear mixing between transforms.
base=lambda z: D(z,off(z))
res=[]
for name,fn in funcs.items():
 for nm,trans in [('id',lambda z:z),('adj+',lambda z:z+np.roll(z,1,axis=1)),('adj-',lambda z:z-np.roll(z,1,axis=1)),('opp+',lambda z:(z+np.roll(z,2,axis=1))[:,:2]),('opp-',lambda z:(z-np.roll(z,2,axis=1))[:,:2]),('pairs',off)]:
  for linear in [True,False]:
   ff=lambda z,k:np.column_stack((base(z) if linear else D(z)[:,:1],fn(trans(z),k)))
   f1=minimize_scalar(lambda k:np.linalg.norm((a:=ff(u,k))@ls(a,v)-v),bounds=(.01,3.3),method='bounded'); a=ff(u,f1.x); e=a@ls(a,v)-v; loo=e/(1-np.sum(a*np.linalg.pinv(a).T,axis=1))[:,None]; res.append((np.std(loo),np.std(e),name,nm,linear,f1.x))
print('v1 alternatives',sorted(res)[:12]);
# Screen downstream models under each plausible calibration using provisional operator recovery.
y=s['train']['y']; tr=s['train']['traj']; target=np.column_stack((y,tr));
for z in zz:
 p=D(z,funcs['tanh'](z,1.48884))@ls(D(u,funcs['tanh'](u,1.48884)),v); q=rf(z,.85)@ls(rf(u,.85),r); b=D(z)@ls(D(s['panel_b']['u']),s['panel_b']['v']); dot=np.sum(p*q,axis=1); cross=(p[:,:,None]*q[:,None,:]).reshape(-1,16)
 for nm,A in [('C',D(p,q,dot)),('C+cross',D(p,q,cross)),('C+B',D(p,q,dot,off(b),b*b,b)),('C+cube',D(p,q,cross,p*p,q*q,off(p),off(q))),('zpoly',D(z,z*z,off(z),z**3)),('B',D(b,b*b,off(b),np.tanh(b)))]:
  pred=A@ls(A[:52],target[:52]); print('spatial',nm,np.round(np.mean(abs(pred[52:]-target[52:]),axis=0)[[0,1,4,8]],4),np.round(np.mean(abs(pred[:52]-target[:52]),axis=0)[[0,1,4,8]],4));

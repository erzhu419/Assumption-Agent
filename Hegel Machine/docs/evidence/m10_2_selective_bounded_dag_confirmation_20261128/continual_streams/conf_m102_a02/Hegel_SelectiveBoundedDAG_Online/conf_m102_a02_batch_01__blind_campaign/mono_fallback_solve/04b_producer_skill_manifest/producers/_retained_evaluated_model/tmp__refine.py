exec(open('/tmp/spatial.py').read().split('fs={}')[0]); np.set_printoptions(precision=5,suppress=True,linewidth=150)
u,v=b['u'],b['v']; print('ranges',x.min(0),x.max(0),cal(s['test']['x']).min(0),cal(s['test']['x']).max(0)); print('telemetry',s['telemetry_vec']); print('summary',tr['summary'])
forms={
'power':(lambda q,a:q*(1+(a[0]*abs(q))**a[1])**a[2],[2.52,11.7,.0358],([.3,2,.0001],[6,30,.5])),
'smoothhinge':(lambda q,a:q*np.logaddexp(0,a[1]*(q*q-a[0])),[.15,18],([.001,.1],[1,100])),
'hinge':(lambda q,a:q*np.maximum(q*q-a[0],0),[.15],([.001],[1.])),
'rootshift':(lambda q,a:q*np.sqrt((q*q-a[0])**2+a[1]),[.15,.01],([0,1e-6],[1,1])),
'rootpower':(lambda q,a:q*((q*q-a[0])**2+a[1])**a[2],[.15,.01,.25],([0,1e-6,.01],[1,1,2])),
'ratcubeplus':(lambda q,a:np.c_[q**3,q**3/(1+a[0]*q*q)],[.2],([.001],[20])),
'rat5':(lambda q,a:q**3/(1+a[0]*q*q+a[1]*q**4),[-.2,.2],([-1.5,0],[2,3])),
'rat5p':(lambda q,a:q**3/(1+a[0]*q*q+a[1]*q**4)**a[2],[-.2,.2,1.],([-1.5,0,.05],[2,3,10])),
'pow46':(lambda q,a:np.c_[q**3,q**5,q**7],[0],([-1],[1])),
'log1p':(lambda q,a:q*np.log1p((a[0]*abs(q))**a[1]),[2.5,10.],([.1,1],[8,40])),
'atan':(lambda q,a:q**3+a[2]*np.arctan(a[0]*q**3)+a[1]*q**5,[2.,0.,-.1],([.01,-5,-5],[20,5,5])),
'gauss':(lambda q,a:q*(q*q+a[1]*np.exp(-a[0]*q*q)),[5.,.1],([.01,-3],[50,3])),
'gauss2':(lambda q,a:q*(1-np.exp(-a[0]*abs(q)**a[1])),[2.,3.],([.01,1],[50,10])),
'cubic_tanh':(lambda q,a:np.tanh(a[0]*q**3),[.4],([.001],[10])),
'sinh':(lambda q,a:np.sinh(a[0]*q),[1.],([.001],[10])),
'cubic_rat':(lambda q,a:q**3/(1+a[0]*abs(q)**6),[.2],([.001],[20])),
'atan_cubic':(lambda q,a:np.arctan(a[0]*q**3),[1.],([.001],[10])),
'cubic_root':(lambda q,a:q**3/(1+a[0]*abs(q)**6)**a[1],[.2,.25],([.001,.01],[20,3])),
}
for name,(fun,z,bounds) in forms.items():
 def feat(q,a):return np.c_[one(q),q,fun(q,a)]
 err,params,op=fitfun(u,v,feat,z,bounds); B=op(x); F=np.c_[one(x),B]; P=np.c_[B[:,0]*B[:,2],B[:,1]*B[:,3]]; C=ls(np.c_[F,P],y); print(name,round(err,6),params,'traj',cv(F)[1::2], 'y paired',cv(np.c_[F,P])[::2], 'y all',cv(poly(B,2))[::2], 'c', C[-2:]);
 if name in ['power','smoothhinge','gauss','cubic_root']:
  hc=ls(F,t); print('traj coef',hc); print('S',np.linalg.svd(hc[1:],compute_uv=False)); print('y coeff',C)

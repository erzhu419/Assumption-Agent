import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): from mixed import *
P=PolynomialFeatures(2)
def diag(z,w):return z*w
F=one(np.c_[a1,a2,a1*a2]); G=one(np.c_[a1,a2,mul(a1,a2)]); print('diag coefficients',fit(F[:52],Y[:52])); print('full coefficients',fit(G[:52],Y[:52]));
for name,z in [('raw',x),('B',b),('C2',a2),('C1',a1)]:
 for name2,h in [('full2',P.fit_transform(z)[:,1:]),('tanh',np.tanh(1.25*z)),('sin',np.sin(1.25*z)),('sat',z/(1+.22*z*z)),('abs',z/(1+.22*abs(z))),('norm',z/(1+.22*np.sum(z*z,axis=1,keepdims=True)))]:
  report('diag plus '+name+' '+name2,np.c_[F[:,1:],h]);
print('direct time fits');
for label,g in [('diag',F),('full',G),('plusC2sq',np.c_[F,P.fit_transform(a2)[:,5:]]),('fullplusC2sq',np.c_[G,P.fit_transform(a2)[:,5:]])]:
 for split in [np.arange(52),np.random.default_rng(30).permutation(72)[:52],np.random.default_rng(50).permutation(72)[:52]]:
  test=np.setdiff1d(np.arange(72),split); p=g[test]@fit(g[split],np.c_[tr['y'],tr['traj']][split]); e=abs(p-np.c_[tr['y'],tr['traj']][test]); print(label,np.r_[e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0)])

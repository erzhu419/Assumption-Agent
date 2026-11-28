exec(open('/tmp/grammar.py').read().split('for name,f in')[0]);
for name,f in [('cos',lambda z,q:1-np.cos(z*q)),('rational',lambda z,q:z*z/(1+q*z*z)),('gaussian',lambda z,q:-np.expm1(-q*z*z)),('tanh2',lambda z,q:np.tanh(q*z)**2),('softabs',lambda z,q:np.sqrt(1+q*z*z)-1),('log',lambda z,q:np.log1p(q*z*z))]:
 def er(p):
  F=aff(u+p[0]*f(u,p[1]));return (F@ls(F,v)-v).ravel()
 fit=least_squares(er,[.5,1.],bounds=([-10,0.01],[25,10])); print(name,fit.x,np.std(fit.fun))
print('times',s['eval_times'],'telemetry',s['telemetry_vec'])

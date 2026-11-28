exec(open('/tmp/panel_cross.py').read().split('for label,T')[0]);
cyc=lambda x:x*np.roll(x,-1,axis=1)
prod=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in pairs])
for pname,P in [('cyc',cyc),('all',prod)]:
 for name,G,box in [('square',lambda x,q:x*x,None),('cube',lambda x,q:x**3,None),('power23',lambda x,q:np.c_[x*x,x**3],None),('sin',lambda x,q:np.sin(q*x),(.05,7)),('cos',lambda x,q:np.cos(q*x),(.05,7)),('atan',lambda x,q:np.arctan(q*x),(.05,7)),('tanh',lambda x,q:np.tanh(q*x),(.05,7)),('softsign',lambda x,q:x/(1+q*abs(x)),(.05,7)),('rational',lambda x,q:x/(1+q*x*x),(.05,7)),('exp',lambda x,q:np.exp(q*x),(-5,5))]:
  cv(pname+' plus '+name,lambda x,q:np.c_[base(x,q),P(x),G(x,q)],box)
  if pname=='cyc' and box is not None: cv(pname+' replaced x '+name,lambda x,q:np.c_[b.one(G(x,q)),P(x)],box)
for name,G in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda x:x/(1+x*x))]:
 for zname,U,V in [('prod xy',lambda x:x,lambda x:np.roll(x,-1,axis=1)),('prod yx',lambda x:np.roll(x,-1,axis=1),lambda x:x),('prod itself',cyc,lambda x:1)]:
  cv(name+' '+zname,lambda x,q:np.c_[base(x,q),G(q*U(x))*V(x)],(.05,7))
for name,G in [('sin',np.sin),('tanh',np.tanh),('rat',lambda x:x/(1+x*x))]:
 cv(name+' both',lambda x,q:np.c_[base(x,q),G(q*x)*G(q*np.roll(x,-1,axis=1))],(.05,7))

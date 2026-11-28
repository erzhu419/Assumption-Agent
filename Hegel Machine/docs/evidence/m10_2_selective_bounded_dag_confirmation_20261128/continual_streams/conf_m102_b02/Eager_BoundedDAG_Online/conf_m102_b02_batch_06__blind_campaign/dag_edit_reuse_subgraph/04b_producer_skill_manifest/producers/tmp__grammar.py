exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
p=s['panel_c'];u=p['u']; v=p['v2']
for name,h in {'quad':lambda u:u+.45*u*u,'exp':lambda u:np.expm1(.9*u)/.9,'cubic':lambda u:np.c_[u,u*u,u**3],'quartic':lambda u:np.c_[u,u*u,u**3,u**4],'sinquad':lambda u:np.c_[u,u*u,np.sin(u)],'tanhquad':lambda u:np.c_[np.tanh(u),u*u,u]}.items():
 A=D(h(u));w=old.fit(A,v);print(name,np.std(A@w-v),w)
A=lambda p:D(sum(q*u**(j+1) for j,q in enumerate(np.r_[1.,p])))
for d in (1,2,3,4,5):
 q=least_squares(lambda p:(A(p)@old.fit(A(p),v)-v).ravel(),np.zeros(d));print('C2 shared series',q.x,np.std(q.fun))

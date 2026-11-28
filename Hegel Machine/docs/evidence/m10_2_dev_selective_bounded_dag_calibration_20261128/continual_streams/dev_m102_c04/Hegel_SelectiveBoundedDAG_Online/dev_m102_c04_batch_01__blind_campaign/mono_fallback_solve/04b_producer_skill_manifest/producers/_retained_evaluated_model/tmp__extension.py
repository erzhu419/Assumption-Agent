
def _graph_kernels(study, coords, n):
    q,z,c,d,cz,dz=coords
    panel=study['panel_c']; pb=study['panel_b']; a0,a1=np.abs(study['telemetry_vec'][:2]); a0=max(a0,.01); a1=max(a1,.001)
    ix=((0,1),(0,3),(1,2),(2,3))
    def quadratic(p): return _aff(np.stack([p[:,i]*p[:,j] for i,j in ix],axis=1))
    w=_ls(quadratic(panel['u']),panel['v1']); v=_ls(_aff(np.cos(2.4*panel['u'])),panel['v2']); b=_ls(_aff(pb['u']),pb['v'])
    B=lambda p:_aff(p)@b
    C=lambda p:quadratic(p)@w
    D=lambda p:_aff(np.cos(2.4*p))@v
    def standard(x): return (x-x[:n].mean(0))/np.maximum(x[:n].std(0),1e-10)
    def gram(x):
        x=standard(x); return x@x.T/x.shape[1]
    def powers(x,k): return np.stack([np.prod(x[:,i],axis=1) for i in combinations_with_replacement(range(4),k)],axis=1)
    groups=[]
    def add(group): groups.append(np.mean(group,axis=0))
    add([gram(powers(x,k)) for x in (q,z) for k in (5,6)])
    fns=(lambda p:np.sin(a0*p),lambda p:np.tanh(a0*p),lambda p:p/(1+a1*p*p),lambda p:p+a1*p*p,lambda p:np.sin(a0*p)+a1*p*p,lambda p:np.cos(a0*p)+a1*p)
    for f in fns:
        p=f(q); u=B(p)
        add([gram(x) for x in (C(p),D(p),C(u),D(u))]);add([gram(powers(x,k)) for x in (p,u,f(z)) for k in (2,3)])
    levels=(c,d,cz,dz,C(d),D(c),C(dz),D(cz))
    for x in (q,z):
        for j in (0,4):
            aug=levels[j:j+4]; gx=gram(x)
            add([gx*gram(p) for p in aug]);add([gram(p*x) for p in aug]);add([gram(x/(1+a1*p*p)) for p in aug]);add([gram(p/(1+a1*x*x)) for p in aug]);add([gram(np.sin(a0*(x+p))) for p in aug]);add([gram(np.cos(a0*(x-p))) for p in aug]);add([gram(powers(p,2)) for p in aug]);add([gram(powers(p,3)) for p in aug]);
    for p,r in ((c,d),(cz,dz),(c,dz),(cz,d),(q,z)):
        add([gram(p)*gram(r),gram(p*r)]);add([gram(p/(1+a1*r*r)),gram(r/(1+a1*p*p))]);add([gram(np.sin(a0*(p+r))),gram(np.cos(a0*(p-r)))])
    for levels0 in ((q,z),(c,d),(cz,dz),(q,c,d),(q,cz,dz)):
        g=sum(gram(p) for p in levels0)/len(levels0); ds=np.maximum(g.diagonal()[:,None]+g.diagonal()[None,:]-2*g,0)
        for k in (0.3,0.8,1.6): groups.append(np.exp(-k*ds));
    return np.asarray(groups)

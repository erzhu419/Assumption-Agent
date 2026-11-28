exec(open('/tmp/profile01.py').read().split('ids=np.arange')[0])
ids=np.arange(len(y)); splits=[(ids[:52],ids[52:])]+[(ids[~np.isin(ids,v)],v) for v in np.array_split(np.random.default_rng(24).permutation(len(y)),6)]
def D(x,p,m):
    h=x@p[:4]; g=x@p[4:8]; j=x@p[8:12]; A=base(x)
    if m=='sq': C=x*h[:,None]**2
    if m=='cube': C=h**3
    if m=='prod': C=h*g*j
    if m=='prod2': C=x*(h*g)[:,None]
    if m=='sq2': C=np.column_stack([x*h[:,None]**2,x*g[:,None]**2])
    if m=='quart': C=affine(x)*h[:,None]**3
    if m=='pure': C=np.column_stack([h**2,h**3])
    if m=='qg': C=quad(x)*h[:,None]
    return np.column_stack([A,C])
for m,n in [('sq',4),('cube',4),('prod',12),('prod2',8),('sq2',8),('quart',4),('pure',4),('qg',4)]:
    es=[]
    for t,v in splits:
        def res(p):
            A=D(q[t],np.pad(p,(0,12-n)),m);return A@fit(A,y[t])-y[t]
        rng=np.random.default_rng(143)
        r=min((least_squares(res,p,max_nfev=120,ftol=1e-7) for p in rng.normal(size=(10,n))),key=lambda r:r.cost)
        X=D(q,np.pad(r.x,(0,12-n)),m); es.append(np.mean(abs(X[v]@fit(X[t],y[t])-y[v])))
        if len(es)==1: print(m,'public train RMSE/params',np.std(r.fun),r.x,flush=True)
    print(m,np.round(es,5),'worst',max(es),flush=True)

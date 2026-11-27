exec(open('/tmp/wall_probe.py').read().split("print('wall coefficients")[0])
# Each boundary row has inward coupling, opposite-wall coupling, and net leakage.
def transport(params):
    aa=[]
    for j,(base,c,r,face,vel) in enumerate(axes):
        a=base.copy(); n=a.shape[0]
        for e,(row,neighbor,opposite) in enumerate([(0,1,n-1),(n-1,n-2,0)]):
            inward,wrap,leak=params[6*j+3*e:6*j+3*e+3]
            a[row,neighbor]=-inward*c[e]; a[row,opposite]=-wrap*c[e]
            a[row,row]=(inward+wrap)*c[e]+leak*r
        aa.append(a.tocsc())
    L=kron(eye(ny),aa[0],format='csc')+kron(aa[1],eye(nx),format='csc')+V
    return np.vstack([splu(L+mu*I).solve(rhs)[ij] for mu in s['mu_shifts']]).T
opts=[]
for j,(base,c,r,face,vel) in enumerate(axes):
    d='xy'[j]; k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(len(k)-1)
    wrapface=(k[0]+k[-1])/(2*h*h); advin=np.maximum([-v[0],v[-1]],0)/h; advwrap=np.maximum([v[0],-v[-1]],0)/h
    opts.append({'robin':np.c_[np.ones(2),np.zeros(2),np.ones(2)],'sealed':np.c_[np.ones(2),np.zeros(2),np.zeros(2)],'reflect':np.c_[2*np.ones(2),np.zeros(2),np.zeros(2)],'periodic_fvm':np.c_[(face+advin)/c,(wrapface+advwrap)/c,np.zeros(2)],'periodic_diffusion':np.c_[face/c,wrapface/c,np.zeros(2)],'periodic_endpoint':np.c_[np.ones(2),np.ones(2),np.zeros(2)],'periodic_endpoint_upwind':np.c_[1+advin/c,1+advwrap/c,np.zeros(2)],'robin_upwind':np.c_[(face+advin)/c,np.zeros(2),np.ones(2)]})
C=[]
for a,b in product(*opts):
    p=np.r_[opts[0][a].ravel(),opts[1][b].ravel()]; e=transport(p)-Y; C.append((max(rms(e)),a,b,p))
print('top exact candidates',[(r,a,b) for r,a,b,p in sorted(C,key=lambda q:q[0])[:8]])
for j in range(len(Y)):
    winner=min(C,key=lambda q:np.linalg.norm((transport(q[-1])-Y)[np.arange(len(Y))!=j])); print('discrete held-out',j,winner[1:3],rms(transport(winner[-1])-Y)[j])
for j in [None,0,1,2]:
    keep=np.array([i for i in range(len(Y)) if i!=j]); start=min(C,key=lambda q:np.linalg.norm((transport(q[-1])-Y)[keep]))[-1]
    for bounds in [(0.,8.),(np.tile([0.,0.,-0.5],4),np.full(12,8.))]:
        fit=least_squares(lambda p:(transport(p)-Y)[keep].ravel(),start,bounds=bounds,gtol=1e-12,xtol=1e-12,ftol=1e-12,max_nfev=160)
        e=transport(fit.x)-Y
        print('held out',j,'min leak',np.min(bounds[0]),'coefficients',np.round(fit.x.reshape(4,3),6),'pulse RMS',rms(e),'shift RMS',np.sqrt(np.mean(e.reshape(3,-1,32)**2,axis=(0,2))))

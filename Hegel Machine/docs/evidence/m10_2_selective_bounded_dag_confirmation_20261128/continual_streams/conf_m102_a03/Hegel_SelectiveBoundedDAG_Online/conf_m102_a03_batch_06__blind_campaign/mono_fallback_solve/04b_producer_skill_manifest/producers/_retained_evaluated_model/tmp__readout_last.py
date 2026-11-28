def sine_branch(panel,z):
    u,v=[np.asarray(panel[k],float) for k in ('u','v')]
    def basis(x,h):return np.c_[affine(x),np.sin(h*x)]
    def res(h):
        A=basis(u,h[0]);return (A@fit(A,v)-v).ravel()
    h=min((least_squares(res,[i],bounds=(.4,6.5)) for i in (1.5,2.8,4.3,5.8)),key=lambda r:r.cost).x[0]
    return basis(z,h)@fit(basis(u,h),v),h


def readout_matrices(z,b,B,f,n):
    def norm(x):return (x-x[:n].mean(0))/np.maximum(x[:n].std(0),1e-8)
    a=norm(z); b=norm(b); B=norm(B); h=norm(np.sin(f*z)); c=norm(z-.4*np.sin(f*z)); H=np.c_[a,h]; Z=np.c_[a,B]; W=np.c_[a,b]; C=np.c_[a,c]; one=np.ones((len(a),len(a)))
    ans={}
    for j,X in enumerate((Z,W,H,C,B,a)):
        p=X.shape[1]; G=X@X.T/p; G1=one+G; d=((X[:,None,:]-X[None,:,:])**2).mean(2)
        for degree in (1,2,3):
            for l in (2.,6.,16.,35.):
                for mix in (.6,3.,15.): ans[j,degree,l,mix]=G1**degree+mix*np.exp(-d/l)
    for j,D in enumerate((h,c,B,b)):
        mixed=((a[:,None,:]-a[None,:,:])**2+(D[:,None,:]-D[None,:,:])**2)*.5
        for l in (.3,1.,3.,8.):
            k=np.exp(-mixed/l); p=np.mean(k,axis=2); p2=(k.sum(2)**2-(k*k).sum(2))/12; p3=(k.sum(2)**3-3*k.sum(2)*(k*k).sum(2)+2*(k**3).sum(2))/24
            G=one+(a@a.T+D@D.T)/8
            for m in (.1,1.,10.):
                ans[6+j,1,l,m]=G+m*(p+.2*p2+.02*p3)
                ans[6+j,2,l,m]=G**2+m*(p+p2+.2*p3)
    return ans

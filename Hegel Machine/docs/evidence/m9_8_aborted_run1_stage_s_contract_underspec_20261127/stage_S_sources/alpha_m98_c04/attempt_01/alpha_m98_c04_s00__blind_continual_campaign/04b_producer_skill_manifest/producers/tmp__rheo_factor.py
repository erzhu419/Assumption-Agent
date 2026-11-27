exec(open('/tmp/rheo_cv.py').read().split('for mode in [')[0])
from scipy.optimize import least_squares

def prep(g,z):
    l,u=g['lab_id'],g['stratum']; hold=g['hold_channel_raw']@b[1:]+b[0]
    cell=(l[:,None]*3+u[:,None]==np.arange(12)).astype(float)
    return hold,cell

def models(g,z,mode):
    hold,cell=prep(g,z); one=np.ones((len(z),1)); base=bw[2]+z@bw[3:]
    columns=[one,z,cell]; prior=[.1]+[.08]*4+[.12]*12
    if mode=='offset': columns=[one,cell]; prior=[.1]+[.12]*12
    if mode=='load': columns += [(hold[:,None]-3),cell*(hold[:,None]-3)]; prior += [.045]+[.035]*12
    if mode=='gain': columns += [cell*(base[:,None]-.3)]; prior += [.25]*12
    return np.column_stack(columns)*prior

def factor(ix, mode, spec):
    g={k:v[ix] for k,v in pi.items()}; z=x[2][ix]; n=len(z)
    D=models(g,z,mode); U=np.column_stack((D,np.eye(n)*.022)); Q=(U[:,None,:]*(h/sc)[None,:,None]).reshape(-1,U.shape[1]); yy=((s._obs(g)-(bw[2]+z@bw[3:])[:,None]*h)/sc).ravel()
    def obj(p,ret=False):
        tau=np.sort(np.exp(p)); R=(s._design(g,z,b,dt,t,tau)[:,:,:2]/sc[None,:,None]).reshape(-1,2)
        if spec=='fixed':
            A=Q; y=yy-R@bw[:2]; prec=np.ones(A.shape[1]); center=np.zeros(A.shape[1]); regtau=0
        else:
            A=np.column_stack((R,Q)); y=yy; prec=np.r_[[1/.06**2,1/.25**2],np.ones(Q.shape[1])]; center=np.r_[bw[:2],np.zeros(Q.shape[1])]; regtau=np.sum(((p-np.log(bt))/[.12,.35])**2)
        matrix=np.diag(prec); rhs=prec*center
        beta=np.linalg.solve(A.T@A+matrix,A.T@y+rhs)
        for _ in range(20):
            r=A@beta-y; w=1/(1+(r/2.5)**2); new=np.linalg.solve(A.T@(w[:,None]*A)+matrix,A.T@(w*y)+rhs)
            if np.linalg.norm(new-beta)<1e-7: beta=new; break
            beta=new
        if ret: return tau,bw[:2] if spec=='fixed' else beta[:2],beta[:D.shape[1]] if spec=='fixed' else beta[2:2+D.shape[1]]
        return np.sum(6.25*np.log1p(((A@beta-y)/2.5)**2))+np.dot(prec,(beta-center)**2)+regtau
    p=np.log(bt)
    if spec!='fixed': p=min((minimize(obj,q,bounds=[(np.log(.05001),np.log(24.999))]*2,method='L-BFGS-B') for q in (p,np.log([.65,7.]))),key=lambda v:v.fun).x
    return obj(p,True)

def pred(g,z,par,mode):
    tau,w,c=par
    return s._design(g,z,b,dt,t,tau)[:,:,:2]@w+(bw[2]+z@bw[3:]+models(g,z,mode)@c)[:,None]*h
cell=pi['lab_id']*3+pi['stratum']; allix=np.ones(len(cell),bool)
for mode,spec in [('offset','fixed'),('offset','soft'),('cell','fixed'),('cell','soft'),('load','soft'),('gain','soft')]:
    pars=factor(allix,mode,spec); print(mode,spec,'full',np.round(np.r_[pars[0],pars[1]],4),'correction range',np.round((models(pi,x[2],mode)@pars[2]).min(),4),np.round((models(pi,x[2],mode)@pars[2]).max(),4),flush=True)
    for tags in (np.arange(len(cell)),cell,pi['lab_id']):
        yp=np.zeros_like(s._obs(pi))
        for k in np.unique(tags):
            ix=tags==k; yp[ix]=pred({key:v[ix] for key,v in pi.items()},x[2][ix],factor(~ix,mode,spec),mode)
        r=abs(yp-s._obs(pi)); c=r[:,:-1]; v=r[:,-1]
        print('folds',len(np.unique(tags)),'mean',np.round([c.mean(),v.mean()],4),'p90',np.round([np.quantile(c,.9),np.quantile(v,.9)],4),'worst cell mean',np.round([max(c[cell==i].mean() for i in np.unique(cell)),max(v[cell==i].mean() for i in np.unique(cell))],4),flush=True)
print('bench params',bt,bw)
R=s._obs(pi)-source_predict(pi,x[2],bt,bw); L=x[2]@bw[3:]+bw[2]; H,_=prep(pi,x[2]); sl=np.median(np.column_stack((R[:,:-1]/t,R[:,-1])),axis=1)
for i in np.argsort(cell): print('bench discrepancy cell',cell[i],np.round([sl[i],L[i],H[i]],4))

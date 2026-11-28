
def _panel_variants(study,z):
    from itertools import combinations_with_replacement
    def bank(x):
        z0=[x,x*x,x**3]
        for w in (.85,1.35,1.8,2.4,3.2):
            z0.extend([np.sin(w*x),np.tanh(w*x),np.arctan(w*x),np.cos(w*x)])
        z0.append(np.column_stack([np.prod(x[:,j],axis=1) for d in (2,3) for j in combinations_with_replacement(range(4),d)]))
        return np.column_stack(z0)
    ans=[]
    for p,k,default in [('panel_b','v','sin'),('panel_c','v1','atan'),('panel_c','v2','quad')]:
        u=study[p]['u']; y=study[p][k]; X=bank(np.vstack((u,z))); n=len(u)
        if k=='v1':
            # Add coupled signed projections and bounded interaction terms.
            xx=np.vstack((u,z)); q=np.column_stack([xx[:,i]+sgn*xx[:,j] for i in range(4) for j in range(i+1,4) for sgn in (-1,1)]); X=np.column_stack([X,*[f(w*q) for f in (np.sin,np.cos,np.tanh,np.arctan) for w in (1.,1.8,2.4)],*[xx[:,i:i+1]/(1+w*xx*xx) for i in range(4) for w in (.5,1.,2.)]])
        X-=X[:n].mean(0); X/=np.maximum(X[:n].std(0),1e-8); yc=y-y.mean(0)
        base=_map_coordinates(study[p],z,default,k); pred=[]
        for j in range(4):
            r=yc[:,j].copy(); R=X[:n].copy(); Q=np.ones((n,1))/np.sqrt(n); sel=[]; opts=[]
            for step in range(min(12,n//2)):
                norm=np.linalg.norm(R,axis=0); norm[sel]=np.inf; ix=np.argmax(abs(R.T@r)/np.maximum(norm,1e-10));
                if norm[ix]<1e-8 or not np.isfinite(norm[ix]): break
                qq=R[:,ix]/norm[ix]; sel.append(ix); Q=np.column_stack((Q,qq)); r-=qq*(qq@r); R-=np.outer(qq,qq@R)
                s=np.mean(abs(r)/np.maximum(1-(Q*Q).sum(1),.03))*(1+.045*step)
                opts.append((s,tuple(sel)))
            ss=min(opts)[1]; pred.append(X[n:,ss]@_linear_fit(X[:n,ss],yc[:,j])+y[:,j].mean())
        ans.append((base,np.column_stack(pred)))
    return ans


def _enhanced(study,z,raw,original,n):
    from itertools import combinations_with_replacement
    y=np.asarray(study['train']['y'],float).reshape(-1); tr=np.asarray(study['train']['traj'],float)
    _,_,V=np.linalg.svd(tr,full_matrices=False); t=tr@V[:2].T; yt=np.column_stack((t,y))
    panels=_panel_variants(study,z)
    B,C,D=[p[0] for p in panels]; E,F,G=[p[1] for p in panels]
    def poly(x,deg): return np.column_stack([np.prod(x[:,j],axis=1) for d in range(1,deg+1) for j in combinations_with_replacement(range(x.shape[1]),d)])
    def standard(x): return (x-x[:n].mean(0))/np.maximum(x[:n].std(0),1e-8)
    zs=standard(z); r2=((zs[:,None]-zs[None,:])**2).mean(2)
    simple=[]
    for b,c,d in ((B,C,D),(E,F,G),(E,C,G),(B,F,D)):
        simple.extend([b,c,d,b+c*d,b+c+d,b+(c-d)**2,b+c*c-d*d,b+np.tanh(c)*d,b+c*np.tanh(d),b/(1+d*d)+c,b/(1+c*c)+d,b*np.tanh(c)+d,b*np.tanh(d)+c,b+c*d/(1+c*c),b+c*d/(1+d*d),b+c/(1+d*d),b+d/(1+c*c),b+c*np.roll(d,2,axis=1),b+np.roll(c,2,axis=1)*d])
        for g in (.5,1.,2.,4.):
            for p,q in ((c,d),(d,c),(b,d),(b,c)):
                simple.append((p+q)/2+np.tanh(g*(p-q))*(p-q)/2)
            simple.append(b+np.tanh(g*np.sum(c*d,axis=1,keepdims=True))*d)
    b0,p0,f0,c0=_basis_bank(z,raw,study)
    polys=[poly(np.column_stack((z,B,C,D))[:,:8],2),poly(z,3),poly(z,4),poly(raw,3),poly(np.column_stack((C,D)),2),poly(np.column_stack((F,G)),3)]
    extras=[]
    for a,b in ((B,C),(B,D),(C,D),(E,F),(E,G),(F,G),(z,F),(z,G)):
        extras.extend([(a[:,:,None]*b[:,None,:]).reshape(len(z),16),a*np.tanh(2*b),a/(1+b*b),b/(1+a*a)])
    dictionary=np.column_stack([*p0,*f0,*c0,*simple,*polys,*extras]); dictionary=standard(dictionary)
    keep=np.max(np.abs(dictionary),axis=0)>1e-7; dictionary=dictionary[:,keep]
    rng=np.random.default_rng(48); splits=np.array_split(rng.permutation(n),5)
    candidates=[]
    def evaluate(design):
        x=standard(design); kernel=x@x.T/max(x.shape[1],1)
        kernels(kernel)
    def kernels(kernel):
        vals=np.logspace(-6,.4,11)
        mu=kernel[:n,:n].mean(0); K=kernel[:n,:n]-mu[:,None]-mu+mu.mean()
        cross=kernel[:,:n]-kernel[:,:n].mean(1,keepdims=True)-mu+mu.mean()
        ev,Q=np.linalg.eigh(K); ev=np.maximum(ev,0); yc=yt-yt.mean(0); pr=Q.T@yc
        for lam in vals:
            iv=1/(ev+lam); dual=Q@(iv[:,None]*pr); diag=(Q*Q)@iv-1/(n*lam)
            loo=dual/np.maximum(diag[:,None],1e-8); mae=np.mean(abs(loo),axis=0)
            candidates.append((mae,cross@dual+yt.mean(0),yt-loo))
    for d in [*simple,*polys,np.column_stack([z,B,C,D,E,F,G]),np.column_stack(extras),dictionary,np.column_stack([*polys,*extras]),np.column_stack([*polys,*simple])]:
        evaluate(d)
    for eps in (.05,.12,.25,.5,1.): kernels(np.exp(-eps*r2))
    # Re-run sparse selection inside every fold, rather than scoring the same chosen supports used for fitting.
    def path(rows,target,mode,bank,max_steps):
        A=bank[rows].copy(); offset=A.mean(0); A-=offset; rr=target[rows]-target[rows].mean(0); Q=np.empty((len(rows),0)); sel=[]; out=[]
        for step in range(max_steps):
            norm=np.linalg.norm(A,axis=0); norm[sel]=np.inf; merit=(A.T@rr)/np.maximum(norm[:,None],1e-10)
            score=abs(merit[:,mode]) if mode<3 else np.linalg.norm(merit/np.maximum(target[rows].std(0),.025),axis=1)
            j=np.argmax(score)
            if not np.isfinite(norm[j]) or norm[j]<1e-8: break
            q=A[:,j]/norm[j]; sel.append(j); A-=np.outer(q,q@A); rr-=np.outer(q,q@rr)
            out.append(np.array(sel))
        return out
    for d in (dictionary,standard(np.column_stack([z,*polys,*extras]))):
        for mode in (0,1,2,3):
            mx=12; cv=np.empty((mx,n,3)); full=[]
            for hold in [*splits,np.array([],dtype=int)]:
                rows=np.setdiff1d(np.arange(n),hold); ps=path(rows,yt,mode,d,mx)
                if not ps: break
                for j in range(mx):
                    ids=ps[min(j,len(ps)-1)]; target_rows=yt[rows]; b=np.column_stack((np.ones(len(rows)),d[np.ix_(rows,ids)])); beta=_linear_fit(b,target_rows)
                    if len(hold): cv[j,hold]=beta[0]+d[np.ix_(hold,ids)]@beta[1:]
                    else: full.append(beta[0]+d[:,ids]@beta[1:])
            for j,p in enumerate(full):
                score=np.mean(abs(cv[j]-yt),axis=0); candidates.append((score,p,cv[j]))
    # Prefer several independently supported predictions over one selected feature set.
    errs=np.array([c[0] for c in candidates]); out=original.copy()
    for j in (0,1,2):
        top=np.argsort(errs[:,j])[:20]; score=errs[top,j]; W=np.exp(-15*(score/score[0]-1)); W/=W.sum()
        pred=W@np.stack([candidates[i][1][:,j] for i in top]); out[:,j]=.35*original[:,j]+.65*pred
    return out


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; train=study['train']['x']; test=study['test']['x']
    x=np.concatenate((train,test)); z=_undo_sensor(x,_fit_sensor(study['panel_a']))
    coords=_fit_indices(z,x,study,len(train))
    coords=_enhanced(study,z,x,coords,len(train))
    return {'c':coords[:len(train)],'query':coords[len(train):]}

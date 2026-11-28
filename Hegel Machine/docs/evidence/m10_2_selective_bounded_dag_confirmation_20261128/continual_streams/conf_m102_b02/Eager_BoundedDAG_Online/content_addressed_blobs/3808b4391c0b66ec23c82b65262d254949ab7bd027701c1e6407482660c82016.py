import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import erf


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-9)[0]


def _panel_a(study, x, verbose=False):
    p = study['panel_a']
    u, v = p['u'], p['v']
    keep = np.linalg.norm(u-v, axis=1) < 3
    u,v = u[keep],v[keep]
    # Diagonal and opposite-coordinate couplings share frequencies across axes.
    def parts(z, q, j):
        h = z[:, (j+2)%4]
        return _design(np.column_stack((z[:,j], np.sin(q[0]*z[:,j]), h, np.sin(q[1]*h))))
    def err(q):
        return np.concatenate([F@_fit(F,v[:,j])-v[:,j] for j in range(4) for F in [parts(u,q,j)]])
    opt = least_squares(err,[2.45,2.45],bounds=([.5,.5],[4,4]))
    q=opt.x
    co=np.stack([_fit(parts(u,q,j),v[:,j]) for j in range(4)])
    def forward(z):
        return np.column_stack([parts(z,q,j)@co[j] for j in range(4)])
    z=x-co[:,0]
    for _ in range(35):
        r=forward(z)-x
        J=np.zeros((len(z),4,4))
        for j in range(4):
            k=(j+2)%4
            J[:,j,j]=co[j,1]+co[j,2]*q[0]*np.cos(q[0]*z[:,j])
            J[:,j,k]=co[j,3]+co[j,4]*q[1]*np.cos(q[1]*z[:,k])
        d=np.linalg.solve(J, r[:,:,None])[:,:,0]
        z-=np.clip(d,-.2,.2)
    if verbose: print('panel A',q, np.sqrt(np.mean(opt.fun**2)),np.max(abs(forward(z)-x)),co)
    return z


def _panel_b(study,z):
    p=study['panel_b']; u,v=p['u'],p['v']
    def residual(q):
        f=_design(np.cos(q*u)); return np.mean((f@_fit(f,v)-v)**2)
    grid=np.linspace(.4,7,23); i=int(np.argmin([residual(q) for q in grid]))
    q=minimize_scalar(residual,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded').x
    return _design(np.cos(q*z))@_fit(_design(np.cos(q*u)),v)




def _panel_c(study):
    p = study['panel_c']; u = p['u']; i,j = np.triu_indices(4,1)
    def train(y, func, q0, inter):
        def design(x,q):
            return _design(np.column_stack((x,func(q*x), *([x[:,i]*x[:,j]] if inter else [x*x]))))
        def err(q):
            f=design(u,q); return np.mean((f@_fit(f,y)-y)**2)
        q=minimize_scalar(err,bounds=q0,method='bounded').x
        c=_fit(design(u,q),y)
        return lambda x: design(x,q)@c
    return train(p['v1'],np.tanh,(.7,2.7),True), train(p['v2'],np.cos,(2,3),False)


def _estimate(s, debug=False):
    from scipy.linalg import cho_factor, cho_solve
    tr=s['train']; n=len(tr['x']); x=np.vstack((tr['x'],s['test']['x']))
    z=_panel_a(s,x); b=_panel_b(s,z); h,g=_panel_c(s)
    traj=np.asarray(tr['traj']); U=np.linalg.svd(traj,full_matrices=False)[2][:2]; y=np.column_stack((tr['y'],traj@U.T)); sy=np.maximum(y.std(0),1e-7); Y=(y-y.mean(0))/sy
    ids=np.eye(n); ix,jx=np.triu_indices(4,1); records=[]; eye=np.eye(n)
    def normalize(f):
        f=f-f[:n].mean(0); return f/np.maximum(f[:n].std(0),1e-5)
    def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
    def evaluate(F,name,types=(0,1,2),lams=(.000006,.0004,.008)):
        F=normalize(F); L=(F@F[:n].T)/F.shape[1]
        for typ in types:
            if typ==0: K=1+L
            elif typ==1: K=(1+L)**2
            elif typ==2: K=(1+L)**3
            elif typ==3:
                norms=np.sum(F*F,axis=1)/F.shape[1]; d=np.maximum(norms[:,None]+norms[None,:n]-2*L,0);K=(1+L)**2+np.exp(-.6*d)
            for lam in lams:
                try: C=cho_factor(K[:n]+lam*eye,check_finite=False); inv=cho_solve(C,ids,check_finite=False); alpha=inv@Y
                except (ValueError,np.linalg.LinAlgError):continue
                residual=alpha/np.diag(inv)[:,None]; error=np.mean(abs(residual),axis=0)
                records.append((error, name, typ, lam, K[n:]@alpha, Y-lam*alpha, residual))
    base={'z':z,'b':b,'x':x,'hz':h(z),'gz':g(z),'hb':h(b),'gb':g(b)}
    for w in (-1,-.6,-.3,-.15,.15,.3,.6,1):
        a=b+w*z; k=f'b{w:+g}z';base[k]=a
        if abs(w) in (.3,.6,1): base['h'+k]=h(a);base['g'+k]=g(a)
    for w in (-1,-.5,.5,1):
        base[f'hb{w:+g}gb']=base['hb']+w*base['gb'];base[f'b{w:+g}hb']=b+w*base['hb'];base[f'b{w:+g}gb']=b+w*base['gb']
    stats=np.asarray(s.get('telemetry_vec',[2,2,2,2])); qs=np.unique(np.r_[.5,.8,1.1,1.4,1.7,2,2.4,2.6,3.1,3.6,4.2,5.,6.,stats[(stats>.4)&(stats<7)]])
    forms={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'erf':erf,'rat':lambda x:x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x)}
    for name,a in base.items():
        evaluate(a,name,(0,1,2,3))
        evaluate(np.c_[a,a**2,a**3,a[:,ix]*a[:,jx]],name+'/poly',(0,1))
        evaluate(np.c_[a,a**2,a**3,a**4,a[:,ix]*a[:,jx],a[:,ix]**2*a[:,jx],a[:,ix]*a[:,jx]**2],name+'/tensor',(0,))
        for fn,f in forms.items():
            for q in qs:
                w=f(q*a); tag=f'{name}/{fn}/{q:g}'
                evaluate(np.c_[a,w],tag,(0,1,2),(.00003,.003))
                evaluate(np.c_[a,a*a,w,w*w,cross(a,w)],tag+'/mixed',(0,),(.000008,.0007))
                evaluate(w,tag+'/only',(0,1),(.00004,.002))
    pairs=[('b','z'),('b','hb'),('b','gb'),('z','hb'),('z','gb'),('hb','gb'),('hz','gz'),('b','hz'),('b','gz'),('z','hz'),('z','gz')]
    for w in (.15,.3,.6,1):
        for prefix in ('','h','g'):
            k=f'{prefix}b{w:+g}z'; l=f'{prefix}b{-w:+g}z';
            if k not in base:continue
            pairs.extend(((k,l),('b',k),('z',k)))
    for ka,kb in pairs:
        a,c=base[ka],base[kb]; name=ka+'|'+kb
        evaluate(np.c_[a,c],name,(0,1,2,3))
        evaluate(np.c_[a,c,a*a,c*c,cross(a,c)],name+'/mixed',(0,1,2))
        for q in (.8,1.4,2.15,2.6,3.1,4,5):
            for fn,f in forms.items():
                w=f(q*a); v=f(q*c); tag=f'{name}/{fn}/{q:g}'
                evaluate(np.c_[a,c,w,v],tag,(0,1,2),(.00004,.003))
                evaluate(np.c_[a,c,w,v,a*a,c*c,cross(w,v)],tag+'/interaction',(0,),(.00004,.003))
                evaluate(np.c_[w,v,cross(w,v)],tag+'/pure',(0,1),(.00004,.003))
    err=np.array([r[0] for r in records]); yp=[]; tp=[]
    for j in range(Y.shape[1]):
        best=np.argsort(err[:,j]); s0=err[best[0],j]; chosen=[]; seen=set()
        for k in best:
            e,name,typ,lam,test,train,resid=records[k]
            if e[j]>s0*1.3 or len(chosen)>=18:break
            # Do not let nearby frequency or ridge variants dominate the ensemble.
            family='/'.join(v for v in name.split('/') if not v.replace('.','').isdigit())
            if family in seen:continue
            seen.add(family); chosen.append(k)
        wt=np.exp(-9*(err[chosen,j]/max(s0,1e-9)-1));wt/=wt.sum()
        tp.append(sum(w*records[k][4][:,j] for k,w in zip(chosen,wt)))
        yp.append(sum(w*records[k][5][:,j] for k,w in zip(chosen,wt)))
        if debug: print('mode',j,'PRESS',s0*sy[j],[(records[k][1:4],round(err[k,j]*sy[j],5)) for k in best[:7]],'ensemble',len(chosen),flush=True)
    return np.vstack((np.column_stack(yp),np.column_stack(tp))), np.c_[z,b], n


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; p,a,n=_estimate(study)
    f=np.column_stack((1.35*p,.075*a));
    return {'train_features':f[:n], 'test_features':f[n:]}

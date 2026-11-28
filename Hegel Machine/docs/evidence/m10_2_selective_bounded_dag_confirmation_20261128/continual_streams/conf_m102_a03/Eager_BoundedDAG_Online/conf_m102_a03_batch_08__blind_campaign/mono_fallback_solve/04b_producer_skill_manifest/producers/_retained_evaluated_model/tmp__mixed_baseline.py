"""Panel-calibrated scalar and trajectory prediction."""
import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-11)[0]


def _roll(x):
    return np.roll(x, -1, axis=1)


def _coords(study, x):
    u, v = [np.asarray(study['panel_a'][k], float) for k in ('u','v')]
    good = np.linalg.norm(v, axis=1) < 5
    u, v = u[good], v[good]
    def features(z):
        return np.stack((np.ones_like(z), z, _roll(z), z*_roll(z), z**3), axis=2)
    a = features(u)
    c = np.stack([_ls(a[:, j], v[:, j]) for j in range(4)], axis=1)
    z = np.array(x, dtype=float, copy=True)
    for _ in range(28):
        delta = np.sum(features(z)*c.T, axis=2) - x
        jac = np.eye(4)[None,:,:]*(c[1]+c[3]*_roll(z)+3*c[4]*z*z)[:,:,None]
        jac += np.roll(np.eye(4), 1, axis=1)[None,:,:]*(c[2]+c[3]*z)[:,:,None]
        z -= np.clip(np.linalg.solve(jac, delta[...,None])[...,0], -.5, .5)
    return z


def _poly(h, degree):
    return np.column_stack([np.ones(len(h))] + [np.prod(h[:, idx], axis=1) for d in range(1,degree+1) for idx in combinations_with_replacement(range(h.shape[1]),d)])


def _panel_b(study, z):
    u, v = [np.asarray(study['panel_b'][k], float) for k in ('u','v')]
    def f(x, w):
        return np.c_[np.ones(len(x)), x, np.sin(w*x)]
    w = least_squares(lambda w: (f(u,w)@_ls(f(u,w),v)-v).ravel(), [2.9], bounds=(1., 4.)).x
    return f(z,w)@_ls(f(u,w),v)


from itertools import combinations
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _readout_features(study, z, n):
    groups = {}
    def add(name, a):
        a = np.array(a, dtype=float, copy=True).reshape(len(z), -1)
        a -= a[:n].mean(0)
        a /= np.maximum(np.sqrt(np.mean(a[:n] ** 2, axis=0)), 1e-9)
        groups[name] = a
    def expand(name, a):
        add(name+'L', a)
        add(name+'Q', a*a)
        add(name+'X', np.column_stack([a[:,i]*a[:,j] for i,j in combinations(range(4),2)]))
    def cross(a,b):
        return (a[:,:,None]*b[:,None,:]).reshape(len(z),-1)
    h = _panel_b(study, z)
    h = h-h[:n].mean(0)
    expand('H',h)
    expand('Z',z)
    add('HZ',cross(h,z))
    coeff = _ls(np.c_[np.ones(n),z[:n]],h[:n]); r = h-coeff[0]-z@coeff[1:]
    r = r@np.linalg.svd(r[:n],full_matrices=False)[2].T
    expand('R',r); add('RZ',cross(r,z))
    for tag, a in [('H',h),('Z',z)]:
        add(tag+'D3',_poly(a,3)[:,15:])
        add(tag+'T',np.column_stack([np.prod(a[:,i],axis=1) for i in combinations(range(4),3)]))
    # Compact auxiliary operator bases estimated without any downstream labels.
    for key in ('v1','v2'):
        u,v = [study['panel_c'][k] for k in ('u',key)]
        if key=='v1':
            def f(x): return np.c_[np.ones(len(x)),x,np.arctan(2.43*x)]
        else:
            def f(x): return np.c_[np.ones(len(x)),x,x*x,np.cos(2.569*x)]
        c=_ls(f(u),v)
        a=f(z)@c; b=f(_panel_b(study,z))@c
        a-=a[:n].mean(0); b-=b[:n].mean(0)
        for suffix, val in [('Z',a),('H',b)]:
            tag=key+suffix
            expand(tag,val); add(tag+'joint',cross(val,h))
    for tag,g in [('rat',z**3/(1+.65*z*z)),('sin',z-.4*np.sin(2.89*z)),('soft',np.tanh(1.25*z))]:
        expand(tag,g); add(tag+'mix',cross(z,g))
    # Mixed-basis polynomial kernels keep higher-order interactions shrunk as a group.
    for tag,a in [('A',np.c_[groups['HL'],groups['ZL']]),('B',groups['HL']),('C',np.c_[groups['HL'],groups['v1ZL'],groups['v2ZL']])]:
        for power in (2,3,4):
            a0=a/np.sqrt(a.shape[1]); add(tag+'p'+str(power),_poly(a0,power)[:,_poly(a0,power-1).shape[1]:])
    return groups


def _predict(groups, Y, n, validation=None):
    names=list(groups)
    J=np.array([a@a[:n].T/a.shape[1] for a in groups.values()]); I=np.eye(n)
    nall=J.shape[1]; K0=np.ones((nall,n)); fits=[]
    def add_fit(K, reg, label, columns):
        cf=cho_factor(K[:n]+reg*I,check_finite=False)
        A=cho_solve(cf,Y[:,columns],check_finite=False)
        d=np.diag(cho_solve(cf,I,check_finite=False))
        e=A/d[:,None]; fits.append((columns,np.mean(abs(e),axis=0),K[n:]@A,label))
    basic=['HL','HQ','HX','ZL','ZQ','ZX','HZ']
    families=[basic,basic[:-1]+['RZ','RQ','RX'],basic+['HD3','HT','ZT'],basic+['v1ZL','v1ZQ','v1ZX','v2ZL','v2ZQ','v2ZX'],basic+['v1HL','v1HQ','v1HX','v2HL','v2HQ','v2HX'],basic+['ratL','ratQ','ratX','ratmix'],basic+['sinL','sinQ','sinX','sinmix'],['HL','HQ','HX','ZL'],['HL','HQ','HX','ZL','Ap2','Ap3'],['HL','HQ','HX','ZL','Cp2','Cp3'],['HL','HQ','HX','ZL','Bp3','Bp4']]
    for kk in families:
        for strength in (.07,.3,1.,3.):
            weights=np.r_[np.ones(3),np.full(len(kk)-3,strength)]
            K=np.einsum('g,gij->ij',weights,J[[names.index(k) for k in kk]])+K0
            for reg in (1e-6,1e-5,1e-4,.0005,.002,.01):
                add_fit(K,reg,('ridge',tuple(kk),strength,reg),list(range(Y.shape[1])))
    def optimize(Ks, yy, loss, labels, cols, init):
        G=np.ascontiguousarray(Ks[:,:n]).reshape(len(Ks),-1); GA=np.ascontiguousarray(Ks).reshape(len(Ks),-1)
        def fun(p):
            w=np.exp(p); M=(w[:-1]@G).reshape(n,n)+K0[:n]+w[-1]*I
            cf=cho_factor(M,check_finite=False); ai=cho_solve(cf,I,check_finite=False); al=ai@yy
            if loss=='mle':
                f=(yy@al+2*np.log(np.diag(cf[0])).sum())/n
                dM=(ai-np.outer(al,al))/n
            else:
                d=np.diag(ai); e=al/d; scale=np.sqrt(e*e+1e-5); q=e/scale/n
                f=scale.mean(); dM=-np.outer(ai@(q/d),al)+(ai*(q*e/d))@ai
            gr=np.r_[G@dM.ravel(),np.trace(dM)]*w
            return f,gr
        pars=np.r_[np.full(len(Ks),init),-6.]; pars[:min(3,len(Ks))]=-2.
        res=minimize(fun,pars,method='L-BFGS-B',jac=True,bounds=[(-18,4)]*len(Ks)+[(-14,0)],options={'maxiter':280,'ftol':2e-10})
        w=np.exp(res.x); K=(w[:-1]@GA).reshape(nall,n)+K0
        add_fit(K,w[-1],(loss,labels,init),cols)
    target=Y[:,1:]; _,_,vh=np.linalg.svd(target,full_matrices=False)
    for k in [0,Y.shape[1]-1]:
        cols=[0] if k==0 else list(range(1,Y.shape[1])); yy=Y[:,0] if k==0 else target@vh[0]/np.sqrt(target.shape[1])
        for loss in ('mle','loo'):
            for init in (-4.,-8.):
                optimize(J,yy,loss,'groups',cols,init)
        sparse_names=basic+['RL','RQ','RX','RZ','HT','ZT','ratL','ratmix','sinmix','v1ZL','v1ZX','v2ZL','v2ZX','v1HL','v2HL']
        features=np.column_stack([groups[g] for g in sparse_names]).T
        Ksp=features[:,:,None]*features[:,None,:n]
        for loss in ('mle','loo'):
            optimize(Ksp,yy,loss,'individual',cols,-6.)
    ans=[]
    for cols in [[0],list(range(1,Y.shape[1]))]:
        buffer=[]
        for original,loo,p,tag in fits:
            if not all(k in original for k in cols):continue
            ix=[original.index(k) for k in cols]; buffer.append((float(np.mean(loo[ix])),p[:,ix],tag))
        buffer.sort(key=lambda r:r[0]); elite=[]; seen=set()
        for e,p,tag in buffer:
            family=tag[:2]
            if family in seen:continue
            seen.add(family); elite.append((e,p,tag))
            if len(elite)>=8:break
        ee=np.array([v[0] for v in elite]); w=np.exp(-(ee-ee[0])/max(.065*ee[0],1e-4)); w/=w.sum()
        p=np.einsum('i,ijk->jk',w,np.array([v[1] for v in elite])); ans.append(p)
        if validation is not None:
            print('columns',cols,'ensemble LOOCV candidates and validation MAEs',[(round(e,6),round(np.mean(abs(q-validation[:,cols])),6),tag) for e,q,tag in elite]);print('ensemble gate',np.mean(abs(p-validation[:,cols])))
    return np.column_stack(ans)


def solve_batch(study):
    tr, te = study['train'], study['test']
    n=len(tr['x']); z=_coords(study,np.concatenate((tr['x'],te['x']),axis=0))
    target=np.column_stack((tr['y'],tr['traj']))
    pred=_predict(_readout_features(study,z,n),target,n)
    y,t=pred[:,0],pred[:,1:]
    return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}

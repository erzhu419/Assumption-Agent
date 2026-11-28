import importlib.util, pickle, numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, least_squares
p=importlib.util.spec_from_file_location('batch','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; tm=s['eval_times']; shift,gain,B,f1,f2=m._operators(s)

def candidates(x,n):
    nrm=lambda q:(q-q[:n].mean(0))/np.maximum(q[:n].std(0),1e-8)
    a=(x-shift)/gain; b=B[0]+a@B[1:]; bi=np.linalg.solve(B[1:].T,(a-B[0]).T).T
    g=nrm(x); u=g@g.T/4
    base=[np.ones_like(u)]+[u**k for k in range(1,4)]
    def groups(q):
        w=nrm(q); C=w@w.T/4
        D=(w[:,None]-w[None,:])**2
        return [C,C**2,np.exp(-.15*D.sum(-1)),np.exp(-.5*D.sum(-1)),np.exp(-1.5*D.sum(-1))]
    def product_features(w):
        q=nrm(w); v=[nrm(f(w)) for f in (lambda q:np.tanh(1.7*q),lambda q:np.logaddexp(1.1*q,-1.1*q), lambda q:np.arctan(2.3*q))]
        Q=q@q.T/4
        K=[h@h.T/4 for h in v]; return [*K,*(Q*h for h in K),*(Q**2*h for h in K)]
    def panel(w):
        u,v=f1(w),f2(w); z=nrm(np.c_[w,u,v]); h=z@z.T/z.shape[1]
        return groups(u)+groups(v)+[h**j for j in (2,3,4)]
    return {k:np.asarray(base+extra) for k,extra in {'raw_iso':groups(x), 'a_warp':panel(a), 'ab_warp':panel(b), 'abi_warp':panel(bi), 'raw_product':product_features(a), 'ab_product':product_features(b), 'both_warp':panel(a)+panel(b), 'mixed_warp':panel(a)+groups(x)+product_features(b)}.items()}

def temporal(z):
    def basis(p):
        r,d,w=p
        return np.c_[tm*np.exp(-r*tm),np.exp(-d*tm)*np.sin(w*tm)]
    def res(p):
        H=basis(p); return (H@m._lstsq(H,z.T)-z.T).ravel()
    opt=least_squares(res,[0,.2,1.1],bounds=([-.4,0,.7],[.5,1,1.7]),max_nfev=100)
    return np.linalg.qr(basis(opt.x))[0].T

folds=[np.arange(56,72),np.arange(16),np.arange(28,44),np.argsort(np.linalg.norm(x,axis=1))[-16:],np.random.default_rng(76).choice(72,16,replace=False),np.random.default_rng(174).choice(72,16,replace=False)]
metrics={}
for j in folds:
    i=np.setdiff1d(np.arange(len(x)),j); f={**s,'train':{**t,**{k:t[k][i] for k in ('x','y','traj')}},'test':{'x':x[j]}}
    r=m.solve_batch(f); ys={'reference':r['y_hat']}; zs={'reference':r['state_pred']}; K=candidates(x[np.r_[i,j]],len(i))
    y,z=t['y'][i],t['traj'][i]; ym,ysd=y.mean(),y.std(); zm=z.mean(0)
    sd=z.std(0); V=np.linalg.svd((z-zm)/sd,full_matrices=False)[2][:2]; T=((z-zm)/sd)@V.T; ts=T.std(0)
    W=temporal(z); P=(z-zm)@W.T; ps=P.std(0)
    for k,kk in K.items():
        ys[k]=m._predict(kk,((y-ym)/ysd)[:,None])[:,0]*ysd+ym
        zs[k]=(m._predict(kk,T/ts)*ts)@V*sd+zm
        zs[k+'_time']=(m._predict(kk,P/ps)*ps)@W+zm
    for k in K:
        ys[k+'_blend']=.5*ys[k]+.5*ys['reference']; zs[k+'_blend']=.5*zs[k]+.5*zs['reference']; zs[k+'_time_blend']=.5*zs[k+'_time']+.5*zs['reference']
    for k,v in ys.items():metrics.setdefault(('y',k),[]).append(np.abs(v-t['y'][j]).mean())
    for k,v in zs.items():
        e=np.abs(v-t['traj'][j]); metrics.setdefault(('z',k),[]).append(e.mean()); metrics.setdefault(('end',k),[]).append(e[:,-1].mean())
for typ in ('y','z','end'):
    print(typ)
    sub={k:a for (tt,k),a in metrics.items() if tt==typ}
    for k in ['reference']+sorted(sub,key=lambda k:(max(sub[k])+np.mean(sub[k])))[:8]:
        a=sub[k]; print(k,np.round(a,5),'mean',round(np.mean(a),5),'worst',round(max(a),5))

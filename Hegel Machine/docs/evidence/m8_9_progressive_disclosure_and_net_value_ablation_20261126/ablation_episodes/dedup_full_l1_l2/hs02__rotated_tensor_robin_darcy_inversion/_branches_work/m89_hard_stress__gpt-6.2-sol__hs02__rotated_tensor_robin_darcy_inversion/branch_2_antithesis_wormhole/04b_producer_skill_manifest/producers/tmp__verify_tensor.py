import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from tensor_darcy_robin import _assemble, _responses, _fit_sources, infer_tensor_darcy_sources
with open('/holdout_task/public_tensor_cases.pkl','rb') as f:
    cases=pickle.load(f)['cases']

def balance(p,u):
    xx,xy,yy=(p[k] for k in ('k_xx','k_xy','k_yy'))
    nx,ny=u.shape; hx,hy=p['Lx']/nx,p['Ly']/ny
    g,amb=p['gamma_robin'],p['u_ambient']
    out=hx*hy*p['sigma']*u
    def v(i,j): return u[min(max(i,0),nx-1),min(max(j,0),ny-1)]
    for i in range(nx):
        for j in range(ny):
            for d in (-1,1):
                k=i+d
                if 0<=k<nx:
                    out[i,j]+=hy/hx*2/(1/xx[i,j]+1/xx[k,j])*(u[i,j]-u[k,j])-d*(xy[i,j]+xy[k,j])/8*(v(i,j+1)-v(i,j-1)+v(k,j+1)-v(k,j-1))
                else:
                    out[i,j]+=hy/(1/g+hx/(2*xx[i,j]))*(u[i,j]-amb)
                k=j+d
                if 0<=k<ny:
                    out[i,j]+=hx/hy*2/(1/yy[i,j]+1/yy[i,k])*(u[i,j]-u[i,k])-d*(xy[i,j]+xy[i,k])/8*(v(i+1,j)-v(i-1,j)+v(i+1,k)-v(i-1,k))
                else:
                    out[i,j]+=hx/(1/g+hy/(2*yy[i,j]))*(u[i,j]-amb)
    return out

rng=np.random.default_rng(489)
fail=[]
for n,c in enumerate(cases):
    p,g=c['problem'],c['gt']; i,j=p['sensor_ij'].T
    r=_responses(p); d=np.c_[r[i,j,1:],np.ones(len(i))]; target=p['y_obs']-r[i,j,0]
    w=g['source_weights']; off=g['sensor_offset']; u=g['u_true']; n_s=len(i); alpha=p['tikhonov_alpha']
    def metrics(coef):
        return [np.linalg.norm(coef[:-1]-w)/np.linalg.norm(w),abs(coef[-1]-off),np.max(abs(r[...,0]+r[...,1:]@coef[:-1]-u))]
    z=infer_tensor_darcy_sources(p)
    print(n,'public [relative weight, abs offset, max field]',metrics(np.r_[z['source_weights'],z['sensor_offset']]), 'known-source forward max',np.max(abs(r[...,0]+r[...,1:]@w-u)))
    clean=abs(p['y_obs']-u[i,j]-off)<0.1
    cv=[]
    for k in range(n_s):
        train=np.arange(n_s)!=k
        coef=_fit_sources(d[train],target[train],alpha)
        cv.append([*metrics(coef),abs(d[k]@coef-target[k])])
    cv=np.array(cv)
    print(n,'LOSO worst',cv[:,:3].max(0),'clean CV RMSE/max',np.sqrt(np.mean(cv[clean,3]**2)),cv[clean,3].max())
    trial=[]; disagreement=[]
    leverage=np.sum(np.linalg.qr(d)[0]**2,axis=1)
    for t in range(200):
        k=round(n_s*.12)
        idx=rng.choice(n_s,k,replace=False) if t else np.argsort(leverage)[-k:]
        y=d@np.r_[w,off]+rng.normal(0,0.001,n_s)
        y[idx]+=rng.uniform(1.5,4,len(idx))*rng.choice([-1,1],len(idx))
        coef=_fit_sources(d,y,alpha)
        keep=np.ones(n_s,dtype=bool); keep[idx]=False
        ridge=np.sqrt(alpha)*np.eye(d.shape[1])[:-1]
        oracle=np.linalg.lstsq(np.vstack((d[keep],ridge)),np.r_[y[keep],np.zeros(len(ridge))],rcond=None)[0]
        trial.append(metrics(coef)); disagreement.append(np.max(abs(coef-oracle)))
    print(n,'12% contamination worst',np.max(trial,axis=0),'max clean-oracle difference',max(disagreement))
    if not (cv[:,0].max()<.10 and cv[clean,3].max()<.008 and np.max(trial,axis=0)[0]<.12 and max(disagreement)<.05): fail.append(('sensor',n))

for shape,sign,gamma in [((7,13),-1,.03),((17,6),1,180.),((1,9),-1,2.),((10,1),1,2.)]:
    nx,ny=shape
    xx=np.exp(rng.uniform(-2,2,shape)); yy=np.exp(rng.uniform(-2,2,shape)); xy=sign*.85*np.sqrt(xx*yy)
    p=dict(Lx=1.7,Ly=.72,k_xx=xx,k_yy=yy,k_xy=xy,sigma=rng.uniform(.1,.4,shape),gamma_robin=gamma,u_ambient=-.65)
    u=rng.normal(size=shape); a,b,area=_assemble(p)
    err=np.max(abs((a@u.ravel()-b).reshape(shape)-balance(p,u)))
    tol=1e-12*max(1,np.max(abs(a@u.ravel())))
    iid=np.array(np.unravel_index(np.arange(nx*ny),shape)).T
    ub=rng.normal(size=(3,nx,ny)); p0=dict(p,u_ambient=0.)
    p['basis']=np.array([balance(p0,q)/area for q in ub]); p['sensor_ij']=iid; p['tikhonov_alpha']=0.
    w=np.array([.9,-1.4,1.2]); off=-.23
    base=_responses(p)[...,0]; truth=base+np.einsum('k,kij->ij',w,ub)
    p['y_obs']=truth.ravel()+off
    worst=np.argsort(np.sum(np.linalg.qr(np.c_[ub.reshape(3,-1).T,np.ones(len(iid))])[0]**2,axis=1))[-max(1,len(iid)//10):]
    p['y_obs'][worst]+=3.
    z=infer_tensor_darcy_sources(p)
    e=np.max(abs(z['predicted_field']-truth))
    print(shape,'sign',sign,'gamma',gamma,'balance max',err,'manufactured field max',e,'weight max',np.max(abs(z['source_weights']-w)),'offset abs',abs(z['sensor_offset']-off))
    assert set(z)=={'source_weights','sensor_offset','predicted_field'} and type(z['sensor_offset']) is float
    assert z['source_weights'].shape==(3,) and z['predicted_field'].shape==shape
    if not (err<tol and np.max(abs(z['source_weights']-w))<1e-8 and abs(z['sensor_offset']-off)<1e-8 and e<1e-8): fail.append(('manufactured',shape))
assert not fail, fail

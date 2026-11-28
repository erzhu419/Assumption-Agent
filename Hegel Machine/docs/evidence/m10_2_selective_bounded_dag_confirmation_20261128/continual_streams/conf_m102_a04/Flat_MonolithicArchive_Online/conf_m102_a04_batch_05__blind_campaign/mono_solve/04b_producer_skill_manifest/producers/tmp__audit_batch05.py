import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('batch_solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; d=s['train']; t=s['eval_times']; y=d['y']; T=d['traj']; x=d['x']; A=s['panel_a']
print('Measured fields:', sorted(s), 'train:', sorted(d), 'test:', sorted(s['test']))
print('Single cohort of continuous responses; no batch labels, class labels, relay state, or stochastic drift observations.')
print('Row spans (rms trajectory, y):', np.sqrt(np.mean(T*T)), np.std(y))
# Isolate calibration error by clipping against the recovered cyclic model and
# comparing affine vs cyclic maps on exactly the same retained panel rows.
u,v=A['u'],A['v']; j=(np.arange(4)+1)%4
f=np.stack([np.ones_like(u),u,u[:,j],u*u[:,j],u**3],axis=-1); keep=np.ones(len(u),bool)
for _ in range(10):
    w=np.stack([m._ls(f[keep,i],v[keep,i]) for i in range(4)])
    e=np.linalg.norm((f*w).sum(-1)-v,axis=1); new=e<=max(5*np.median(e),.0008)
    if np.array_equal(new,keep): break
    keep=new
b=m._ls(m._with_bias(u[keep]),v[keep]); e_aff=np.linalg.norm(m._with_bias(u[keep])@b-v[keep],axis=1)
print('Calibration measured inlier count, max affine, max cyclic:',keep.sum(),e_aff.max(),e[keep].max())
z=m._calibrate(A,x); za=np.linalg.solve(b[1:].T,(x-b[0]).T).T
B=s['panel_b']; zB=m._with_bias(z)@m._ls(m._with_bias(B['u']),B['v'])
def designs(q):
    g,h=m._operators(s['panel_c'],q); outer=(g[:,:,None]*h[:,None,:]).reshape(len(q),-1); dot=(g*h).sum(1)
    base=m._with_bias(np.c_[g,h,dot]); full=m._with_bias(np.c_[g,h,outer]); diag=m._with_bias(np.c_[g,h,g*h]); return g,h,base,full,diag,dot
_,_,ya,ta,_,_=designs(za); _,_,yb,tb,_,_=designs(zB); g,h,Y,F,diag,dot=designs(z)
# Differences span the linear basis already; include squared differences too.
i,j=np.triu_indices(4,1); pairs=np.c_[g[:,i]-g[:,j],h[:,i]-h[:,j],g+h,g-h]; Q=np.c_[dot,g,h,pairs]
variants=[('cyclic full',Y,F),('affine ablation',ya,ta),('panel B insertion',yb,tb),('dot trajectory',Y,Y),('diagonal trajectory',Y,diag)]
# One predeclared nested selection rule for additional algebraic terms. Scale Q
# from the fitting sample only and regularize added terms to avoid rank inflation.
extras=[('pair differences',0,0),('pair squares',0,0)]+[('quotient',a,k) for a in (.1,1.) for k in (0,1,2)]
def pred(spec, a, b, D, target):
    from numpy.linalg import solve
    name,alpha,kind=spec
    if name=='base': return D[b]@m._ls(D[a],target[a])
    if name.startswith('pair'): E=pairs if name=='pair differences' else pairs**2
    else:
        q=Q[:,[0]] if kind==0 else Q[:,1:9] if kind==1 else pairs
        q=q/np.maximum(np.std(q[a],axis=0),1e-8)
        numerator=np.c_[g,h,dot] if kind==0 else np.c_[g,h] if kind==1 else pairs
        E=numerator/(1+alpha*q*q)
    w=m._ls(D[a],E[a]); E=E-D@w
    scale=np.std(E[a],0); use=scale>1e-7
    E=E[:,use]/scale[use]; base=m._ls(D[a],target[a]); r=target[a]-D[a]@base
    coef=solve(E[a].T@E[a]+4*np.eye(E.shape[1]),E[a].T@r)
    return D[b]@base+E[b]@coef
rng=np.random.default_rng(905); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(12)]; splits+=[np.argsort(z[:,j]) for j in range(4)]+[np.argsort(-z[:,j]) for j in range(4)]
R=[]; nested=[]
for k in splits:
    a,b=k[:52],k[52:]; basis=m._temporal(t,T[a]); amp=m._ls(basis,T.T).T
    R.append([(np.mean(abs(Yi[b]@m._ls(Yi[a],y[a])-y[b])),np.mean(abs((Fi[b]@m._ls(Fi[a],amp[a]))@basis.T-T[b]))) for _,Yi,Fi in variants])
    choices=[('base',0,0)]+extras; scores=np.zeros((len(choices),2))
    for bv in np.array_split(rng.permutation(a),4):
        av=np.setdiff1d(a,bv); inner=m._temporal(t,T[av]); inneramp=m._ls(inner,T.T).T
        for i,c in enumerate(choices):
            scores[i]+=[np.mean(abs(pred(c,av,bv,Y,y)-y[bv])),np.mean(abs(pred(c,av,bv,F,inneramp)@inner.T-T[bv]))]
    iy,it=np.argmin(scores,axis=0); yp=pred(choices[iy],a,b,Y,y); tp=pred(choices[it],a,b,F,amp)@basis.T
    nested.append((np.mean(abs(yp-y[b])),np.mean(abs(tp-T[b]))))
    if np.array_equal(k,splits[0]): print('Required split selected augmented alternatives:',choices[iy],choices[it])
R=np.array(R); nested=np.array(nested)
for i,(name,*_) in enumerate(variants): print(name,'required split, nested-audit slice mean, worst:',R[0,i],R[:,i].mean(0),R[:,i].max(0))
print('Nested algebraic challengers: required split, slice mean, worst:', nested[0],nested.mean(0),nested.max(0))
assert R[:,0].max()<.038
out=m.solve_batch(s); yh,sp=out['y_hat'],out['state_pred']; print('Actual test shapes:',yh.shape,sp.shape)
assert yh.shape==(36,) and sp.shape==(36,8) and np.isfinite(yh).all() and np.isfinite(sp).all()
np.testing.assert_array_equal(out['diagnostics_vec'],[yh.mean(),yh.std(),sp[:,-1].mean(),sp[:,-1].std()])

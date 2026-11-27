import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
    from probe_m99 import *
from itertools import combinations
from scipy.linalg import lstsq
p,h=m._loads(g,b)
c,r=m._basis(ta,p,h,gx,dt,t)
y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None]-c@th,g['instant_rate_obs']-r@th]
T=np.vstack((np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1.]))
z=lstsq(T,y.T)[0].T
print('joint temporal residual curve/rate:',np.sqrt(np.mean((y-z@T.T)**2,axis=0)))
u,s,v=np.linalg.svd(z,full_matrices=False); print('correction coefficient singular values and directions:',s,v)
print('corrections, sample:'); print(np.c_[z,h,p,gx][:8])
# Search compact input grammars. Select support on the training coupons only.
base={'hold':h,**{f'p{j}':p[:,j] for j in range(6)},**{f'x{j}':gx[:,j] for j in range(4)}}
gap=r[:,:2]*ta
F={**base,'gap0':gap[:,0],'gap1':gap[:,1],'drift':r[:,2:]@th[2:],'mem0':h-gap[:,0],'mem1':h-gap[:,1]}
for j,k in combinations(base,2):
    F[f'{j}-{k}']=base[j]-base[k]; F[f'{j}+{k}']=base[j]+base[k]
for j,k in combinations([*base],2): F[f'{j}*{k}']=base[j]*base[k]
for j,a in base.items(): F[f'{j}^2']=a*a
F={**F,**{f'abs({k})':np.abs(a) for k,a in list(F.items()) if '*' not in k and '^' not in k},**{f'tanh({k})':np.tanh(a) for k,a in list(F.items()) if '*' not in k and '^' not in k}}
keys=list(F); A=np.column_stack(list(F.values())); labs=g['lab_id'];
for target,n in [(z,'coefficient'),(y,'curve and rate')]:
 for lab in [None,*np.unique(labs)]:
    trn=np.ones(len(labs),bool) if lab is None else labs!=lab; test=~trn if lab is not None else trn
    train=target[trn]; a=A[trn]; mu=a.mean(0); ac=a-mu; scale=np.maximum(np.linalg.norm(ac,axis=0),1e-12); a=ac/scale; center=train.mean(0); e=train-center; support=[]
    for j in range(7):
        if support:
            q=np.linalg.qr(a[:,support])[0]; new=a-q@(q.T@a)
        else: new=a
        score=np.linalg.norm(new.T@e,axis=1)/np.maximum(np.linalg.norm(new,axis=0),1e-12); score[np.linalg.norm(new,axis=0)<1e-7]=-1
        ix=np.argmax(score); support.append(ix); coef=lstsq(a[:,support],train-center)[0]; e=train-center-a[:,support]@coef
        pred=center+((A[test]-mu)/scale)[:,support]@coef
        de=pred-target[test]; de=de@T.T if n=='coefficient' else de
        print(n,lab,j+1,keys[ix],np.round([np.sqrt(np.mean(de[:,:-1]**2)),np.sqrt(np.mean(de[:,-1]**2))],5))
    if lab is None: print('original-units coefficients:',[(keys[k],np.round(v,5).tolist()) for k,v in zip(support,coef/scale[support,None])])

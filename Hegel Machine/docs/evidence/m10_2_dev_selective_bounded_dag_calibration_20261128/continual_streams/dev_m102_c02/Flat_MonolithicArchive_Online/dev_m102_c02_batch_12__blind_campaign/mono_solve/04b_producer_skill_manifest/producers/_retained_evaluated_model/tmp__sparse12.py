exec(open('/tmp/check12.py').read().split('metrics={}')[0])
from itertools import combinations
pairs=list(combinations(range(4),2)); match=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
def design(b,kind):
    if kind=='quadratic': return m._quadratic(b)
    if kind=='affine': return m._affine(b)
    return m._affine(np.column_stack([b,sum(b[:,i]*b[:,j] for i,j in kind)]))
for label,kind in [('quadratic','quadratic'),('paired',None)]:
    errs={}; sel={}
    for key,q in tests.items():
        a=np.setdiff1d(np.arange(len(y)),q)
        if kind is None:
            def loo(k):
                h=design(b[a],k); w=m._fit(h,y[a]); lev=(h*np.linalg.pinv(h).T).sum(1)
                return np.mean(((y[a]-h@w)/(1-lev))**2)
            best=min(['affine',*match],key=loo); sel[str(best)]=sel.get(str(best),0)+1
        else: best=kind
        h=design(b,best); errs[key]=np.mean(abs(h[q]@m._fit(h[a],y[a])-y[q]))
    print(label,'public',errs['public'],'worst',max(errs,key=errs.get),max(errs.values()),'mean',np.mean(list(errs.values())),'selections',sel)
panel=s['panel_b']; u,v=panel['u'],panel['v']; pc=[]
for k in range(len(u)):
    mask=np.arange(len(u))!=k; op=m._operator({'u':u[mask],'v':v[mask]}); pc.append(np.abs(op(u[k:k+1])[0]-v[k]))
print('cosine operator refitted panel LOO MAE',np.mean(pc),'worst row MAE',np.max(np.mean(pc,axis=1)))

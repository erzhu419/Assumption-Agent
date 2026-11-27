import contextlib,io,runpy,itertools,numpy as np
with contextlib.redirect_stdout(io.StringIO()): d=runpy.run_path('/tmp/compare_walls.py')
s=d['s']; assess=d['assess']; responses=d['base']._responses; old=d['axes']; n=d['n']; y=d['y']; z=d['z']; m=d['m']; nr=d['nr']; sr=d['sr']; cap=d['cap']; total=d['total']; qp=d['latest']._box_simplex
rosette=np.tile(np.arange(n)//4,len(s['mu_shifts'])); shifts=np.repeat(np.arange(len(s['mu_shifts'])),n); banks=s['row_bank_id']
ids=(rosette,shifts,banks)
opts=[]
for axis in old:
    more=dict(axis)
    for name,row in [('L',0),('U',-1)]:
        a=axis['N'].tolil(); a[row,row]=axis['R'][row,row]; more[name]=a.tocsc()
    opts.append(more)
rows=[]
for (cx,ax),(cy,ay) in itertools.product(*(p.items() for p in opts)):
    X,_=responses(s,(ax,ay)); errors=[assess(X,idx) for idx in ids]; rows.append((sum(e[0]**2 for e in errors),cx+cy,errors,X))
for _,name,errors,_ in sorted(rows,key=lambda r:r[0])[:8]: print('wall candidate',name,'pooled/worst rosette shift bank',np.round(errors,6).tolist())
X=next(X for _,name,_,X in rows if name=='CN'); errors=[]
for bank in np.unique(banks):
    test=banks==bank; train=~test; Z=z[train]; D=np.column_stack((X[train],y[train])); coeff=np.linalg.solve(Z.T@Z+nr*np.eye(z.shape[1]),Z.T@D); P=D-Z@coeff; A,b=P[:,:m],P[:,m]; w,_=qp(A.T@A+sr*np.eye(m),-A.T@b,cap,total)
    errors.append((int(bank),int(test.sum()),np.mean((y[test]-X[test]@w)**2)**.5,np.mean((y[test]-X[test]@w-z[test]@(coeff[:,m]-coeff[:,:m]@w))**2)**.5))
print('held-out bank (id, count, RMSE ignoring nuisance, RMSE with nuisance)',errors)

exec(open('/tmp/stress_batch.py').read().split('for mode in')[0])
for name,i,j in folds:
    p=m._profile(u[i],np.c_[y[i], z[i] @ m._time_basis(s['eval_times'],z[i])]); g=m._design(u,m._quadratic(u),p)
    res=[]
    for wl,wq in [(0,0),(.01,.1),(.03,.3),(.1,1),(.3,3),(1,8),(0,2),(0,8),(.05,2),(.1,5)]:
        d=np.diag(np.sqrt(np.r_[0,np.repeat(wl,4),np.repeat(wq,10),np.zeros(4)])); c=np.linalg.lstsq(np.r_[g[i],d],np.r_[y[i],np.zeros(len(d))],rcond=None)[0]; res.append(np.mean(abs(g[j]@c-y[j])))
    print(name,np.round(res,5))
    if name=='52/20':
        print('shape',np.exp(p),'coef',np.linalg.lstsq(g[i],y[i],rcond=None)[0]); print('norms',np.linalg.norm(g[i],axis=0))
    if name in ('axis2low','axis3high'):
        c=np.linalg.lstsq(g[i],y[i],rcond=None)[0]; print('range and error',u[j,2 if name=='axis2low' else 3], (g[j]@c-y[j]).round(5))
print('ranges train/test',u.min(0),u.max(0),m._coordinates(s['panel_a'],s['test']['x']).min(0),m._coordinates(s['panel_a'],s['test']['x']).max(0))

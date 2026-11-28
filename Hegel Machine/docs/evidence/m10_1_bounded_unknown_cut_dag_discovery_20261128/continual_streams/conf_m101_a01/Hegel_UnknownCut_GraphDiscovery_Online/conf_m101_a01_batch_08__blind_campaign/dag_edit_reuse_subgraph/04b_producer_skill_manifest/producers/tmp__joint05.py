exec(open('/tmp/index05.py').read().split("for struct in ['linear'")[0])
import importlib.util
sp=importlib.util.spec_from_file_location('candidate','/holdout_task/node_01.py'); b=importlib.util.module_from_spec(sp); sp.loader.exec_module(b)
panel=s['panel_c']; c=b._readout(panel['u'],panel['v2'],q)
d=((q[:,None]-q[None,:])**2).sum(2)+.04*((c[:,None]-c[None,:])**2).sum(2)
K=np.exp(-.5*d); Z=Q[:,1:]; W=Q[:,:5]

def joint(i,j,Y,kind,starts=6):
    w=W[i]; u=np.linalg.qr(w)[0]; yi=Y[i]; yr=yi-u@(u.T@yi)
    f=np.tanh if kind=='tanh' else lambda z:z/np.sqrt(1+z*z)
    def loss(p):
        h=f(p[0]+Z[i]@p[1:]); h-=u@(u.T@h)
        return np.r_[(h[:,None]*(h@yr)/(h@h+1e-10)-yr).ravel(),.002*p]
    cov=np.linalg.lstsq(Q[i],yi,rcond=None)[0]; direction=cov[:,0]
    rng=np.random.default_rng(15)
    init=[direction]+list(rng.normal(0,.8,(starts-1,len(direction))))
    p=min((least_squares(loss,g,max_nfev=90) for g in init),key=lambda a:a.cost).x
    A=np.c_[W,f(p[0]+Z@p[1:])]; coef=np.linalg.lstsq(A[i],yi,rcond=None)[0]
    return A@coef,p

records={}; params=[]
for name,i,j in splits:
    ki=K[np.ix_(i,i)]+.003*np.eye(len(i)); kj=K[np.ix_(j,i)]; ntr=len(i)
    def record(key,pred,target): records.setdefault(key,[]).append(np.mean(abs(pred-target)))
    X=W[i]; tcoef=np.linalg.lstsq(X,T[i],rcond=None)[0]; tr=T-W@tcoef
    U,S,V=np.linalg.svd(tr[i],full_matrices=False); time=V[:2].T
    joint_y=np.column_stack((y, tr@time/np.maximum(np.std(U[:,:2]*S[:2],axis=0),1e-6)*.2))
    for kind in ('tanh','soft'):
        pred,p=joint(i,j,y[:,None],kind,8); base=pred[:,0]; params.append(p)
        correction=kj@np.linalg.solve(ki,y[i]-base[i]); kr=y[i].mean()+K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+.0001*np.eye(ntr),y[i]-y[i].mean())
        record(kind,base[j],y[j]); record(kind+' residual kernel',base[j]+correction,y[j]); record(kind+' blend',.8*base[j]+.2*kr,y[j]); record(kind+' shared',joint(i,j,joint_y,kind)[0][j,0],y[j]);
        tf,_=joint(i,j,T@time,kind); tp=W@tcoef+tf@time.T-(W@tcoef@time)@time.T
        record('traj '+kind,tp[j],T[j]); record('traj '+kind+' blend',.5*tp[j]+.5*Q[j]@np.linalg.lstsq(Q[i],T[i],rcond=None)[0],T[j])
    record('traj quadratic',Q[j]@np.linalg.lstsq(Q[i],T[i],rcond=None)[0],T[j])
for k,v in records.items():
    v=np.array(v); print(k,np.round([v[0],v[1:25].mean(),v[1:25].max(),v[25:].mean(),v[25:].max()],6), 'worst_tail',splits[25+v[25:].argmax()][0])

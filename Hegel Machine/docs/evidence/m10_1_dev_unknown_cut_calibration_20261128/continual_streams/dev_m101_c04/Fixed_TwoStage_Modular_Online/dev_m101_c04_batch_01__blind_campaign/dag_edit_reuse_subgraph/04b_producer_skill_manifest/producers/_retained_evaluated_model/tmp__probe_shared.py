exec(open('/tmp/probe_node01.py').read().split('for w in (u,x):')[0])
from scipy.optimize import least_squares
A=poly(x,2)

def fit(train, cols, kind):
    B=A[:,cols]; D=B[train]; Y=y[train]
    c=np.linalg.lstsq(D,Y,rcond=None)[0]
    if kind=='shared':
        def predict(B,p):
            h=B@p[:-2]
            return h + p[-2]*np.tanh(p[-1]*h)
        sol=least_squares(lambda p:predict(D,p)-Y,np.r_[0.75*c,0.45,1.5],bounds=(np.r_[np.full(len(c),-np.inf),0,0.1],np.r_[np.full(len(c),np.inf),2,8]))
        return predict(B,sol.x),sol.x
    P=np.linalg.pinv(D); R=Y-D@(P@Y)
    def coeff(b):
        F=np.tanh(D@b); G=F-D@(P@F); a=(G@R)/(G@G)
        return P@(Y-a*F),a
    def res(b):
        c,a=coeff(b); return D@c+a*np.tanh(D@b)-Y
    p=np.r_[c[:5]*0.8/np.std(D@c),np.zeros(len(c)-5)]
    sol=least_squares(res,p,max_nfev=200); c,a=coeff(sol.x)
    return B@c+a*np.tanh(B@sol.x),np.r_[c,a,sol.x]

n=len(y); allrows=np.arange(n)
folds=[('public',allrows[52:])]+[(f'block{k}',v) for k,v in enumerate(np.array_split(allrows,6))]
folds += [(f'extreme{k}',np.argsort(s)[-14:]) for k,s in enumerate((np.linalg.norm(x,axis=1),abs(x[:,0]*x[:,2]),abs(x[:,1]*x[:,3])))]
for label,cols,kind in [('independent full',list(range(15)),'independent'),('independent compact',[0,1,2,3,4,7,11],'independent'),('shared full',list(range(15)),'shared'),('shared compact',[0,1,2,3,4,7,11],'shared')]:
    print(label)
    for name,val in folds:
        tr=np.setdiff1d(allrows,val); pred,p=fit(tr,cols,kind); e=abs(pred[val]-y[val]); print(name,round(e.mean(),6),round(e.max(),6), 'coef '+str(p) if name=='public' else '')
    pred,p=fit(allrows,cols,kind); print('full mae/coef',np.mean(abs(pred-y)),p)

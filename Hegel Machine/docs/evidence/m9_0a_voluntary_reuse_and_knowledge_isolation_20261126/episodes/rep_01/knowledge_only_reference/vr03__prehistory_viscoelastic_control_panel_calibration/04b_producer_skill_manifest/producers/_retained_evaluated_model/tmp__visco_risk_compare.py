exec(open('/tmp/visco_audit.py').read().split('for i,d in enumerate(data):')[0])
from scipy.special import expit
from scipy.optimize import minimize
for i,d in enumerate(data):
    co=d['cohort']; t=co['train']
    x=m._aligned_moduli(t,np.asarray(co['taus']))
    c,s,y=[t[k] for k in ('center_id','stratum','y')]
    print('cohort',i)
    for coefficient_ridge,stratum_ridge in [(1/3,0),(1/3,0.5),(1/3,1),(1/3,2),(0.1,0.5),(1,0.5)]:
        out=[]
        for j in np.unique(c):
            h=c==j
            w=m._cell_weights(c[~h],s[~h]); z=x[~h]; ok=np.isfinite(z)
            fill=(w@np.where(ok,z,0))/(w@ok)
            a=np.column_stack((np.ones(len(x)),np.where(np.isfinite(x),x,fill),s[:,None]==[1,2]))
            aa=a[~h]; yy=y[~h]; pen=np.array([0]+[coefficient_ridge]*3+[stratum_ridge]*2)
            def obj(b):
                l=aa@b
                return w@(np.logaddexp(0,l)-yy*l)+0.5*(b*pen)@b,aa.T@(w*(expit(l)-yy))+pen*b
            b=minimize(obj,np.zeros(6),jac=True,method='BFGS').x
            p=expit(a[h]@b)
            out.append((roc_auc_score(y[h],p),np.mean((y[h]-p)**2)))
        print('ridge',coefficient_ridge,stratum_ridge,'worst AUC/Brier',round(min(a for a,b in out),4),round(max(b for a,b in out),4),'LOCO',np.round(out,4).tolist())

from chamber_probe import *
from scipy.optimize import least_squares
q,th=fit(); _,h=load(c); labs=c['lab_id']; base=h[:,None]*(basis(q)@th[:2]); E=yc-base
raw=c['coupon_features_raw']; valid=np.isfinite(raw).all(1); tc=np.r_[t,1.];
def standard(s):
    X=s['coupon_features_raw'].copy(); labs=s['lab_id']; refs=s['ref_coupons']
    for lab in np.unique(labs):
        m=labs==lab; P=refs[m]; X[m]=(X[m]-np.median(P.mean(0)))/np.median(P.std(0))
    return X
X=standard(c)
def impute(s,X,method):
    labs=s['lab_id']; strata=s['stratum']; _,i,n=np.unique(np.c_[labs,strata],axis=0,return_inverse=True,return_counts=True); w=1/n[i,None]; mu=np.nansum(w*X,0)/np.sum(w*np.isfinite(X),0)
    result=X.copy()
    for a,j in zip(*np.where(~np.isfinite(X))):
        masks={'cell':(labs==labs[a])&(strata==strata[a]),'lab':labs==labs[a],'stratum':strata==strata[a],'unweighted':np.ones(len(X),bool)}
        if method in masks:
            v=X[masks[method],j]; result[a,j]=np.nanmean(v) if np.isfinite(v).any() else mu[j]
        else: result[a,j]={'zero':0.,'balanced':mu[j],'raw_balanced':mu[j]}[method]
    if method=='raw_balanced':
        rr=s['coupon_features_raw']; v=np.nansum(w*rr,0)/np.sum(w*np.isfinite(rr),0)
        for a,j in zip(*np.where(~np.isfinite(X))):
            P=s['ref_coupons'][labs==labs[a]]; result[a,j]=(v[j]-np.median(P.mean(0)))/np.median(P.std(0))
    return result
for method in ('zero','balanced','unweighted','cell','lab','stratum','raw_balanced'):
    z=impute(c,X,method); D=th[2]+z@th[3:]; pred=np.empty_like(yc); labs_err=[]
    for lab in np.unique(labs):
        m=(labs!=lab)&valid; test=labs==lab; G=D[m]@E[m]/(D[m]@D[m]); pred[test]=base[test]+D[test,None]*G; labs_err.append(err(yc[test],pred[test])[:2]);
    print(method,'OOF',err(yc,pred)[:2],'worst lab',np.max(labs_err,0),'complete',err(yc[valid],pred[valid])[:2],'missing',err(yc[~valid],pred[~valid])[:2])
D=th[2]+impute(c,X,'zero')@th[3:]; G=D[valid]@E[valid]/(D[valid]@D[valid]); trueD=(E*unit)@(G*unit)/np.sum((G*unit)**2)
print('shape',np.round(G,8),'implied standardized fills',[(i,j,round((trueD[i]-D[i])/th[3+j],5)) for i,j in zip(*np.where(~np.isfinite(X)))]); print('full baseline reset',err(yc,base+D[:,None]*G),'complete',err(yc[valid],(base+D[:,None]*G)[valid])); print('balanced fills',impute(c,X,'balanced')[~np.isfinite(X)]);
for k in ('intrinsic','one','two','rational'):
    def calc(logtau, all=False):
        z=np.exp(logtau)
        if k=='intrinsic': B=np.c_[basis(q),tc]
        elif k=='rational': B=np.c_[np.r_[t/(1+t/z[0]),1.],tc]
        else: B=np.c_[basis(z),tc]
        coeff=np.linalg.lstsq(B*unit[:,None],G*unit,rcond=None)[0]; p=B@coeff
        return (coeff,p) if all else (p-G)*unit
    z={'intrinsic':[1.], 'one':[2.], 'two':[.5,3.], 'rational':[2.]}[k]; opt=least_squares(calc,np.log(z),bounds=(np.log(.08),np.log(30))) if k!='intrinsic' else None
    v=opt.x if opt else np.log(z); coeff,P=calc(v,True); print(k,np.exp(v),coeff,'per time rmse',np.sqrt(np.mean((P-G)**2)), 'complete',err(yc[valid],(base+D[:,None]*P)[valid]))

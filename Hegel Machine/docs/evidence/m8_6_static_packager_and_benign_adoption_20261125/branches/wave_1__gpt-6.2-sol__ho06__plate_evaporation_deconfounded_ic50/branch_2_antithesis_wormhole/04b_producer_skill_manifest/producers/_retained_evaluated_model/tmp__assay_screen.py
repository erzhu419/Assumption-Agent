import pickle
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit
with open('/holdout_task/public_screen.pkl','rb') as f: data=pickle.load(f)
a=[(i,w['row'],w['col'],w['well_type'],w['compound_id'],w['log10_conc_uM'],w['fluorescence']) for i,p in enumerate(data['plates']) for w in p['wells']]
p,r,c,t,ids,x,y=map(np.asarray,zip(*a)); x=x.astype(float)
dm=t=='DMSO'; ki=t=='POS_KILL'; co=t=='COMPOUND'; B=np.array([np.median(y[ki&(p==i)]) for i in np.unique(p)])[p]; med=np.array([np.median((y-B)[dm&(p==i)]) for i in np.unique(p)])[p]; z=(y-B)/med
u=np.minimum(r,15-r); v=np.minimum(c,23-c); d=np.minimum(u,v)
def basis(mode,k):
    q = [np.ones(len(y))] if mode=='scalar' else [np.ones(len(y)),(r-7.5)/7.5,(c-11.5)/11.5]
    if mode=='nearest': q += [np.exp(-d/k)]
    if mode=='gauss': q += [np.exp(-(d/k)**2)]
    if mode=='rational': q += [(1+d/k)**-2]
    if mode=='sides': q += [np.exp(-u/k)+np.exp(-v/k),np.exp(-(u+v)/k)]
    if mode=='maxlin': q += [np.exp(-d/k), (r-7.5)/7.5*np.exp(-d/k), (c-11.5)/11.5*np.exp(-d/k)]
    return np.array(q).T

def calibrate(mode, mask, k=None):
    def do(k):
        M=basis(mode,k); out=np.zeros(len(y)); cost=0.
        for i in np.unique(p):
            s=p==i; m=mask&s; A=M[m]; zz=z[m]
            beta=np.linalg.lstsq(A,zz,rcond=None)[0]
            f=least_squares(lambda b:A@b-zz,beta,loss='soft_l1',f_scale=.012)
            out[s]=M[s]@f.x; cost+=f.cost
        return out,cost
    if k is None: k=minimize_scalar(lambda k:do(k)[1],bounds=(.4,5),method='bounded').x if mode not in ('scalar','linear') else 1.
    return do(k)[0],k

def score(g,sel=None,loss='cauchy'):
    if sel is None: sel=co
    es=[]
    for cid, gt in data['gt'].items():
        m=sel&(ids==cid); xx=x[m]; zz=z[m]; gg=g[m]; jj=np.argmin(abs(zz/gg-.5))
        def f(b): return gg*expit(np.log(10)*b[1]*(b[0]-xx))-zz
        init=least_squares(f,[xx[jj],1.],bounds=([-7,.1],[6,5]),loss='soft_l1',f_scale=.025).x
        fit=least_squares(f,init,bounds=([-7,.1],[6,5]),loss=loss,f_scale=.015).x
        es.append(fit-[gt['log10_ic50_uM'],gt['hill_slope']])
    return np.max(np.abs(es),axis=0),np.argmax(np.abs(es),axis=0)

if __name__=='__main__':
    for mode in ['scalar','linear','nearest','gauss','rational','sides','maxlin']:
        g,k=calibrate(mode,dm); er=z-g
        print(mode,round(k,3),'full',score(g),'plate-jack',np.round(np.max([score(g,co&(p!=i))[0] for i in np.unique(p)],axis=0),4),'training control RMSE',round(np.sqrt(np.mean(er[dm]**2)),5))
        folds=[]
        for idx in np.unique(r[dm]*24+c[dm]):
            test=dm&(r*24+c==idx); gp,_=calibrate(mode,dm&~test); folds.extend(zip(d[test],(z-gp)[test]))
        folds=np.array(folds); print('  control LOO rms by d',[(int(dd),round(float(np.sqrt(np.mean(folds[folds[:,0]==dd,1]**2))),4)) for dd in np.unique(folds[:,0])], 'residual levels',[(int(dd),round(float(np.median(er[dm&(d==dd)])),4)) for dd in np.unique(d[dm])])

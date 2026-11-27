exec(open('/tmp/m98_panel_fit.py').read().split('for l in np.unique(g):')[0])
S=1.; scale=np.r_[np.full(len(t),.013),.024]; y=Y

def predict(p):
    T=tau*np.exp(p[:2]); s=np.exp(p[3]); D=np.exp(p[2])*v
    return np.c_[G@(-np.expm1(-t[None,:]/T[:,None]))-D[:,None]*s*np.expm1(-t/s),G@(1/T)+D]

def metrics(pred):
    e=abs(pred-y)
    return np.round([max(np.median(e[g==l,:-1]) for l in np.unique(g)),max(np.median(e[g==l,-1]) for l in np.unique(g)),max(np.median(e[strata==l,:-1]) for l in np.unique(strata)),max(np.median(e[strata==l,-1]) for l in np.unique(strata)),max(np.median(e[:,:-1],axis=0)),np.quantile(e[:,-2],.9),np.median(e[~bad,:-1]),np.median(e[~bad,-1])],6)
print('unit-amplitude, unit-time saturation with intrinsic branch times:',metrics(predict(np.zeros(4))))
for cols in ([2],[1,2,3],[0,1,2,3]):
    def unpack(z):
        p=np.zeros(4);p[cols]=z;return p
    for groups in (g,strata):
        O=np.empty_like(y); ps=[]
        for l in np.unique(groups):
            a=groups==l
            z=least_squares(lambda z:((predict(unpack(z))-y)[~a]/scale).ravel(),np.zeros(len(cols)),loss='soft_l1',f_scale=1.5).x
            O[a]=predict(unpack(z))[a];ps.append(z)
        print('free indices',cols,'fold parameters',np.round(ps,5),'metrics',metrics(O))
E=predict(np.zeros(4))-y
print('complete per-time signed median',np.round(np.median(E[~bad],axis=0),5)); print('missing rows',np.flatnonzero(bad),'signed median per row',np.round(np.median(E[bad,:-1],axis=1),5),'rate',np.round(E[bad,-1],5))

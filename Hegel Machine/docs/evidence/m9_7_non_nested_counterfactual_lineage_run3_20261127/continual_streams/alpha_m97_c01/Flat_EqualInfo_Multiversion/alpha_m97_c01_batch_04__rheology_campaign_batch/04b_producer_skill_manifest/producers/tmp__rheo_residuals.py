exec(open('/tmp/rheo_diagnose.py').read().split('for label,ar,rob')[0])
ar=(*args[:-1],np.ones(n,bool))
tau,w=fit(ar,yc,yr,True)
c,r=s._design(tau,dt,t,*ar); ec=c@w-yc; er=r@w-yr
print('row lab stratum damage missing median_curve_error signed_curve_residuals rate_residual')
for i in np.argsort(np.median(np.abs(ec),axis=1))[-30:]:
    print(i,labs[i],tr['stratum'][i],round(args[3][i],3),missing[i],round(np.median(np.abs(ec[i])),4),ec[i].round(3),round(er[i],3))
print('curve outliers >3',np.argwhere(np.abs(ec)>3).tolist())
for k in [False,True]:
    m=args[-1]==k
    print('active',k,'curve/rate quantiles',np.quantile(np.abs(ec[m]),[.5,.8,.9,.95,.99]),np.quantile(np.abs(er[m]),[.5,.8,.9,.95,.99]))

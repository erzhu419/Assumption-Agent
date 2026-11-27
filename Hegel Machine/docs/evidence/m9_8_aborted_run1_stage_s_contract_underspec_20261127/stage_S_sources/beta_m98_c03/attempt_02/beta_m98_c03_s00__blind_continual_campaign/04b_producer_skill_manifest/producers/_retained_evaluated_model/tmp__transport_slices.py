from transport_probe import *
B,E,hx,hy=init(1,'ar','face','velocity'); d0=np.repeat([rx/hx,ry/hy],2)
ns=len(ids); group=np.repeat(np.arange(len(shifts)),ns)
for name,W in [('axis',np.repeat(np.eye(2),2,axis=0)),('walls',np.eye(4))]:
    for k in range(len(shifts)):
        use=group!=k
        fit=least_squares(lambda a:(predict(B,E@(d0+W@a))-y)[use].ravel(),np.zeros(W.shape[1]),gtol=1e-12)
        errors=(predict(B,E@(d0+W@fit.x))-y)[~use]
        print(name,'held shift',k,'RMSE per pulse',np.sqrt(np.mean(errors**2,axis=0)))
    fit=least_squares(lambda a:(predict(B,E@(d0+W@a))-y).ravel(),np.zeros(W.shape[1]),gtol=1e-12)
    print(name,'relative coefficients',(d0+W@fit.x)/d0,'Jacobian singular values',np.linalg.svd(fit.jac,compute_uv=False))
    res=fit.fun.reshape(len(shifts),ns,2)
    print('largest residual sensors',[(coords[j].tolist(), round(float(e),8)) for j,e in sorted(enumerate(np.sqrt(np.mean(res**2,axis=(0,2)))),key=lambda v:-v[1])[:8]])

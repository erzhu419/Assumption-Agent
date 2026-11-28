from probe import *
from scipy.spatial.distance import cdist
Y=np.c_[y,T]; ids=np.arange(len(x)); edge=lambda v:np.argsort(v)[-14:]
folds=[ids[56:],*np.array_split(ids,4),edge(np.linalg.norm(x,axis=1)),*[edge(abs(z)) for z in x.T]]
P=poly(x,2); C=poly(x,3)[:,15:]; H0=P@P.T; H3=C@C.T

def kernel(z, kind, l):
    r=cdist(z,z)/l
    if kind=='mat': return (1+r+r*r/3)*np.exp(-r)
    if kind=='rbf': return np.exp(-r*r/2)
    w=np.exp(-.5*((z[:,None,:]-z[None,:,:])/l)**2)
    return sum(w[:,:,i]*w[:,:,j] for i in range(4) for j in range(i,4))/10

def evaluate(K,reg):
    ee=[]; preds=[]
    for jj in folds:
        ii=np.setdiff1d(ids,jj); V=np.linalg.svd(T[ii],full_matrices=False)[2][:2]
        p=K[np.ix_(jj,ii)]@np.linalg.solve(K[np.ix_(ii,ii)]+reg*np.eye(len(ii)),Y[ii]); p[:,1:]=p[:,1:]@V.T@V
        preds.append(p); ee.append(np.abs(p-Y[jj]))
    return np.array([[e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()] for e in ee]),preds

zs={'x':x,'b':x@B[1:],'bi':x@np.linalg.inv(B[1:])}; rs=[]
for zn,z in zs.items():
    z=z/z.std(0)
    for kind in ['mat','rbf','pair']:
        for l in [.9,1.6,2.5,4.]:
            G=kernel(z,kind,l)
            for c in [0,.05,.25,1]:
                for reg in [.0003,.003,.03]:
                    E,_=evaluate(H0+c*H3+G,reg); rs.append(((zn,kind,l,c,reg),E))
for col in [0,1,2]:
    print('criterion',col)
    for cfg,E in sorted(rs,key=lambda r:(r[1].max(0)+r[1].mean(0))[col])[:6]:
        print(cfg,'mean',E.mean(0).round(5),'worst',E.max(0).round(5),'axis4',E[-1].round(5))
# Check whether the trajectory bottleneck comes from a few held-out rows.
_,a=evaluate(H0+kernel(x/x.std(0),'mat',1.6),.003); z=zs['b']; _,b=evaluate(H0+kernel(z/z.std(0),'mat',1.6),.03)
jj=folds[-1]; e=np.abs(.75*a[-1][:,-1]+.25*b[-1][:,-1]-T[jj,-1]); print('axis4 endpoint errors sorted',np.sort(e).round(5))

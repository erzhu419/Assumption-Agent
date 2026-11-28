from graph import *
import pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; f,inv,op,c,d=maps(s); z=inv(t['x']); b=op(z); u=np.cos(2.6*z); H=aff(b)
for label,W in [('b',b),('u',u),('z',z),('c(z)',c(z)),('d(z)',d(z)),('c(b)',c(b)),('d(b)',d(b))]:
    print(label)
    P=[]; names=[]
    for j in range(2,5):
        for i in itertools.combinations_with_replacement(range(4),j):P.append(np.prod(W[:,i],axis=1)); names.append(i)
    P=np.array(P).T
    R=[]; A=H.copy()
    for i in range(8):
        cv=fit(A[:52],y[:52]); r=y[:52]-A[:52]@cv; L=P[:52]-A[:52]@fit(A[:52],P[:52]); sc=abs(r@L)/np.maximum(1e-8,np.linalg.norm(L,axis=0));j=np.argmax(sc); A=np.c_[A,P[:,j]]; cv=fit(A[:52],y[:52]);R.append((names[j],round(np.std(y[:52]-A[:52]@cv),5),round(np.mean(abs(y[52:]-A[52:]@cv)),5)))
    print(R)
    for k in (1,2,3,4):
        A=np.c_[H,W,P[:,[len(j)<=k for j in names]]];cv=fit(A[:52],y[:52]); print(k,round(np.std(y[:52]-A[:52]@cv),5),round(np.mean(abs(y[52:]-A[52:]@cv)),5),end='; ')
    print()
print('b linear',fit(H[:52],y[:52])); A=np.c_[H,np.column_stack([b[:,i]*b[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)])]; print('b quadratic',fit(A[:52],y[:52])); print('pair',[(ij,round(np.mean(abs(y[52:]-np.c_[H,b[:,ij[0]]*b[:,ij[1]]][52:]@fit(np.c_[H,b[:,ij[0]]*b[:,ij[1]]][:52],y[:52]))),5)) for ij in itertools.combinations_with_replacement(range(4),2)])

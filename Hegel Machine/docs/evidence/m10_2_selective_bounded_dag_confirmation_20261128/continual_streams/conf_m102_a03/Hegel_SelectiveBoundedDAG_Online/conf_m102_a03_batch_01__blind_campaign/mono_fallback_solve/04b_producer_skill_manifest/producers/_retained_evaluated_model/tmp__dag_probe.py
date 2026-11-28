import pickle,itertools,numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];Y=np.c_[s['train']['y'],s['train']['traj']];n=52
np.set_printoptions(precision=5,suppress=True,linewidth=150)
def aug(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def rm(z):return np.sqrt(np.mean(z*z,axis=0))
def chk(D):
 p=D[n:]@ls(D[:n],Y[:n]);e=np.abs(p-Y[n:]);r=D[:n]@ls(D[:n],Y[:n])-Y[:n];return np.round([e[:,0].mean(),e[:,1:].mean(),rm(r[:,0]),np.sqrt(np.mean(r[:,1:]**2))],5)
u,v=(s['panel_a'][k] for k in ['u','v']);a=aug(u);g=np.linalg.norm(v-a@ls(a,v),axis=1)<4
p=PolynomialFeatures(3);f=p.fit_transform(u);c=ls(f[g],v[g]);print('A monomials:')
for nm,co in zip(p.get_feature_names_out(),c):
 if max(abs(co))>0.0002:print(nm,co)
for j in range(4):
 h=abs(c[:,j])>0.0002;c[~h,j]=0;c[h,j]=ls(f[g][:,h],v[g,j]);print('A sparse',j,rm(f[g]@c[:,j]-v[g,j]))
q=np.array([least_squares(lambda z:p.transform(z[None,:])[0]@c-row, row,gtol=1e-12).x for row in x]);print('inverse error',abs(p.transform(q)@c-x).max(), 'range',q.min(0),q.max(0))
b=ls(aug(s['panel_b']['u']),s['panel_b']['v']);bq=aug(q)@b;bx=aug(x)@b
for nm,z in [('x',x),('q',q),('bq',bq),('bx',bx)]:
 for deg in [1,2,3]:print(nm,deg,chk(PolynomialFeatures(deg).fit_transform(z)))
 for label,w in [('tanh',np.tanh(z)),('sin',np.sin(z)),('cube',z**3),('qsin',np.sin(1.25*z)),('quad',z**2),('softsign',z/(1+abs(z)))]:print(nm,label,chk(np.c_[aug(z),w]),'alone',chk(aug(w)))
 for label,w in [('same',z[:,0]*z[:,2]+z[:,1]*z[:,3]),('cross',z[:,0]*z[:,2]-z[:,1]*z[:,3])]:print(nm,label,chk(np.c_[aug(z),w]))
np.savez('/tmp/enc.npz',q=q,bq=bq,bx=bx,c=c,b=b)
u=s['panel_c']['u'];p2=PolynomialFeatures(2);P=p2.fit_transform(u);v=s['panel_c']['v2'];print('C2 quad coefs:');print(np.c_[p2.get_feature_names_out(),ls(P,v).round(5).astype(str)])
print('C input distribution',u.min(0),u.max(0),'corr',np.corrcoef(u.T));print('C1 sample',np.c_[u,s['panel_c']['v1']][:7]);print('C2 sample',np.c_[u,v][:7]);print('y train correlation vs inverses',np.corrcoef(np.c_[q,bq,Y[:,0]].T)[-1]);print('trajectory time modes',np.linalg.svd(Y[:,1:],full_matrices=False)[2][:2])

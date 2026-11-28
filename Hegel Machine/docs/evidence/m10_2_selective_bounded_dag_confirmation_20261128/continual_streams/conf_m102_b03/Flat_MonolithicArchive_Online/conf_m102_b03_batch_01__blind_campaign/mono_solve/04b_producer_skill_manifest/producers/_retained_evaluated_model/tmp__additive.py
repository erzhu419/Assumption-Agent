from operators import *
from itertools import product,combinations_with_replacement
from scipy.optimize import least_squares,minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
x=s['train']['x']; z=(x-a)/b; t=s['eval_times']; tr=s['train']['traj']; H=-np.expm1(-t[:,None]*np.array([.38,1.8,5.4])); Y=np.c_[s['train']['y'],fit(H,tr.T).T]; n=52
P=PolynomialFeatures(3).fit_transform(z); Q=P[:,:15]; cub=P[:,15:]; C=s['panel_c']; u=C['u']; cv=lambda u:np.column_stack([u[:,i]*u[:,j] for i,j in [(0,1),(0,3),(1,2),(2,3)]]); A=fit(F(cv(u)),C['v1']); G=fit(F(np.cos(2.4*u)),C['v2']); f=lambda w:F(cv(w))@A; h=lambda w:F(np.cos(2.4*w))@G
W={'z':z,'Bz':Bf(z),'BL':z@B[1:],'Bt':Binv(z),'BT':z@B[1:].T,'x':x}; K={**W,**{k+'F':f(w) for k,w in W.items()},**{k+'H':h(w) for k,w in W.items()}}; fs={'cos':np.cos,'sin':np.sin,'tanh':np.tanh,'rat':lambda v:1/(1+v*v),'sig':lambda v:1/(1+np.exp(-v)),'soft':lambda v:v/(1+abs(v))}; out=lambda v,w:(v[:,:,None]*w[:,None,:]).reshape(len(v),-1)
R=[]
def ck(D,key):
 D=D/np.maximum(.0001,np.std(D[:n],axis=0)); B0=fit(D[:n],Y[:n]); E=abs(Y-D@B0); R.append((E[:n].mean(0),E[n:].mean(0),key));return E[:n].mean(0)
for k,w in K.items():
 for g in np.linspace(.4,5,24):
  for l,fun in fs.items():
   U=fun(w*g)
   for i,D in enumerate([np.c_[P,U],np.c_[Q,U,U*w],np.c_[Q,U,out(z,U)],np.c_[Q,U,out(f(z),U)],np.c_[Q,U,out(f(Bf(z)),U)]]): ck(D,(k,round(g,2),l,i))
for j in range(4):
 print('OUTPUT',j)
 for k in [0,1]:
  for r in sorted(R,key=lambda r:r[k][j])[:8]: print(r[0].round(4),r[1].round(4),r[2]);
print('fast leave split cubic',ck(P,('cubic',)),R[-1][1]);
# Test whether early and late columns share a precisely cubic mode, allowing temporal basis to rotate.
for p in [Q,P]:
 r=tr-p@fit(p,tr);_,ss,vv=np.linalg.svd(r);print('time after polynomial projection',ss,'top mode',vv[:3]);
 print('alignment with relaxation',vv@H)

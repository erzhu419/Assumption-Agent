from refine import *
from sklearn.linear_model import lars_path
from scipy.optimize import least_squares
tr=s['train'];Y=np.c_[tr['y'],tr['traj']]; z,bs,C,D=operators(s,tr['x']); ix=np.arange(52); jx=np.arange(52,72); u=s['panel_b']['u']; v=s['panel_b']['v']
for k,f,p in [('abs',lambda u,p:u*(np.abs(u)**p[0]+p[1]),[1,-.3]),('absplus',lambda u,p:u*(np.abs(u)**p[0]+p[1]+p[2]*u*u),[1,-.3,0]),('signed',lambda u,p:u+p[0]*np.tanh(p[1]*u)*np.abs(u),[-1,1]),('cbrtsin',lambda u,p:u+p[0]*np.sin(p[1]*u)+p[2]*np.cbrt(u),[-.4,2.9,0]),('sinabs',lambda u,p:u+p[0]*np.sin(p[1]*u)+p[2]*u*abs(u),[-.4,2.9,0]),('softdead',lambda u,p:u*(1-p[0]/np.sqrt(1+p[1]*u*u)),[1,3]),('offsetsin',lambda u,p:u+(p[0]+p[2]*abs(u))*np.sin(p[1]*u),[-.4,2.9,0]),('sinr',lambda u,p:u+p[0]*np.sin(p[1]*u)/(1+abs(p[2])*u*u),[-.4,2.9,0])]:
 def calc(p):
  F=aff(f(u,p));return F,ls(F,v)
 o=least_squares(lambda p:(calc(p)[0]@calc(p)[1]-v).ravel(),p,max_nfev=220); bs[k]=aff(f(z,o.x))@calc(o.x)[1];print('panel',k,o.x,np.std(o.fun))
print('dexact',np.std(D(s['panel_c']['u'])-s['panel_c']['v2']))
rows=[]; store={}
def add(k,F):
 store[k]=F
 for r,P,L in fit(F,Y,ix,jx,[.001,.01,.05,.2,1.]):
  v=abs(P-Y[jx]).mean(0);rows.append((L[0]+L[1:].mean(),v[0],v[1:].mean(),k,r))
for bk,b in bs.items():
 c=C(b); d=D(z)
 for ck,cc in [('id',c),('t',np.tanh(1.25*c)),('r',c/(1+.22*c*c)),('a',np.arcsinh(1.25*c))]:
  for dk,dd in [('id',d),('t',np.tanh(1.25*d)),('r',d/(1+.22*d*d)),('a',np.arcsinh(1.25*d))]:
   k=(bk,ck,dk); inter=(cc[:,:,None]*dd[:,None,:]).reshape(len(cc),16)
   add(k+('cross',),np.c_[aff(np.c_[cc,dd]),inter]); add(k+('c2',),np.c_[poly(cc,2),dd,inter]); add(k+('d2',),np.c_[poly(dd,2),cc,inter]); add(k+('full',),np.c_[poly(np.c_[cc,dd],2),c,d]); add(k+('diag',),np.c_[aff(np.c_[cc,dd]),cc*dd,cc**2,dd**2]); add(k+('cc',),np.c_[aff(np.c_[cc,dd]),inter,cc*d,cc*dd**2]);
for i in range(3): print('top',i,*sorted(rows,key=lambda a:a[i])[:10],sep='\n')
for n in (3,8,20,50):
 rr=sorted(rows)[:n]; P=np.mean([next(fit(store[k],Y,ix,jx,[r]))[1] for _,_,_,k,r in rr],0); v=abs(P-Y[jx]).mean(0);print('avg',n,v[0],v[1:].mean())
# Compare the robust shape-preserving readout to the already passing alternatives.
for bk,ck,dk,ft,r in [('polyfull','t','t','full',.001),('odd5','t','t','full',.001),('original','id','t','full',.05)]:
 k=(bk,ck,dk,ft); F=store[k]; print('REF',k,r,next(fit(F,Y,ix,jx,[r]))[1].shape)

from fit_dag05 import *
from scipy.optimize import least_squares
pc=raw['panel_c'];u=pc['u'];v=pc['v1'];w=pc['v2'];
def fit_panel(X,Y,fn,init,lo,hi):
 def design(X,p):return np.column_stack((np.ones(len(X)),fn(X,p)))
 def res(p):
  A=design(X,p);return (A@np.linalg.lstsq(A,Y,rcond=None)[0]-Y).ravel()
 o=least_squares(res,init,bounds=(lo,hi),max_nfev=400);A=design(X,o.x);coef=np.linalg.lstsq(A,Y,rcond=None)[0];print('panel',np.round(o.x,5),score(A,Y),flush=True);return lambda X:design(X,o.x)@coef
odd=lambda X,p:(X+p[0]*X**3)/(1+p[1]*X**2+p[2]*X**4)
odd2=lambda X,p:np.tanh(p[0]*X)+p[1]*X/(1+p[2]*X**2)
even=lambda X,p:np.column_stack((X,X**2/(1+p[0]*X**2)))
V=fit_panel(u,v,odd,[3.65,6.2,2],[-5,0,0],[10,30,30]);V2=fit_panel(u,v,odd2,[2,.1,1],[.1,-3,0],[12,3,25]);W=fit_panel(u,w,even,[.16],[0],[4]);
for coord in [z,V(z),V2(z),W(z)]:print('coords',score(poly(coord,2),y),score(poly(coord,3),y),score(poly(coord,1),T),flush=True)

def trial(base, X, X2, family, tr, te, starts=12):
 B=base[tr];L=X[tr];M=X2[tr];Q=np.linalg.qr(B,mode='reduced')[0];yr=y[tr]-Q@(Q.T@y[tr]);n=L.shape[1]; m=M.shape[1];rng=np.random.default_rng(45)
 def feats(L,M,p):
  a=L@p[:n]+p[n];b=M@p[n+1:n+1+m]+p[-1]
  if family=='sin':return np.column_stack((np.sin(a),np.sin(b)))
  if family=='tanh':return np.column_stack((np.tanh(a),np.tanh(b)))
  if family=='mix':return np.column_stack((np.sin(a),np.tanh(b)))
  if family=='product':return np.column_stack((np.sin(a)*np.cos(b),np.tanh(a)*np.tanh(b)))
  return np.column_stack((a/(1+b*b),1/(1+b*b)))
 def res(p):
  H=feats(L,M,p);H-=Q@(Q.T@H);return H@np.linalg.lstsq(H,yr,rcond=None)[0]-yr
 opt=min((least_squares(res,rng.normal(0,s,n+m+2),max_nfev=170) for s in np.resize([.3,.7,1.2],starts)),key=lambda o:o.cost)
 A=np.column_stack((base,feats(X,X2,opt.x)));coef=np.linalg.lstsq(A[tr],y[tr],rcond=None)[0];print(family,round(np.mean(abs(A[tr]@coef-y[tr])),5),round(np.mean(abs(A[te]@coef-y[te])),5),np.round(opt.x,2),flush=True)
if __name__=='__main__':
 tr=np.arange(52);te=np.arange(52,72)
 for nm,X in [('z',z),('v1',V(z)),('v2',W(z))]:
  for yn,Y in [('v1',V(z)),('v2',W(z))]:
   for bn,B in [('add',np.column_stack((np.ones(len(z)),V(z),W(z)))),('quadr',np.column_stack((poly(V(z),2),W(z))))]:
    print(nm,yn,bn,flush=True)
    for f in ('sin','tanh','mix','product','rat'):trial(B,X,Y,f,tr,te)

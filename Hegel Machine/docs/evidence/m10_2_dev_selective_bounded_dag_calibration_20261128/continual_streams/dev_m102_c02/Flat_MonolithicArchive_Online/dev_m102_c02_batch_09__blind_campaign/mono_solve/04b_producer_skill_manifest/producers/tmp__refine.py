from newgraph import *
u,v=s['panel_a']['u'],s['panel_a']['v']; k=np.linalg.norm(v,axis=1)<4
r=lambda x:np.roll(x,2,axis=1)
def base(x,j,p):
 a,b=p
 if j==0:return x
 if j==1:return x/(1+a*abs(x)**b)
 if j==2:return x*(1-np.exp(-a*abs(x)**b))
 if j==3:return np.sign(x)*abs(x)**b
 if j==4:return x/(1+a*abs(x)**b)**.5
 if j==5:return x+a*np.sign(x)*abs(x)**b
 if j==6:return np.tanh(a*x)
 if j==7:return np.arctan(a*x)
def ac(x,p,i,j):
 g,h=base(x,i,p[:2]),base(r(x),j,p[2:]); eye=np.eye(4)
 return np.concatenate([np.broadcast_to(eye,(len(x),4,4)),g[:,:,None]*eye,h[:,:,None]*eye],axis=2)
L=[]
for i,j in product(range(8),repeat=2):
 def rr(p):
  H=ac(u[k],p,i,j).reshape(-1,12); return H@fit(H,v[k].ravel())-v[k].ravel()
 o=least_squares(rr,[.22,2.15,.8,1.2],bounds=([-.6 if i==5 else .001,.3,-.6 if j==5 else .001,.3],[8,5,8,5]),max_nfev=140);L.append((np.std(o.fun),i,j,o.x))
L.sort(key=lambda row:row[0]);print('cal models',L[:9]);_,i,j,p=L[0]; H=ac(u[k],p,i,j).reshape(-1,12); coef=fit(H,v[k].ravel());print(coef)
z=np.array([least_squares(lambda z:(ac(z[None],p,i,j)@coef)[0]-q,q,gtol=1e-12).x for q in tr['x']]);b=Bf(z);c=C(b);d=D(b)
print('z',z.min(0),z.max(0)); print('baseline',errs(aff(np.c_[b,d,a._quad(b),a._quad(d)])),errs(aff(np.c_[b,b**3])))
np.savez('/tmp/refine.npz',z=z,b=b,c=c,d=d,cal_p=p,cal_ij=[i,j],cal_c=coef)

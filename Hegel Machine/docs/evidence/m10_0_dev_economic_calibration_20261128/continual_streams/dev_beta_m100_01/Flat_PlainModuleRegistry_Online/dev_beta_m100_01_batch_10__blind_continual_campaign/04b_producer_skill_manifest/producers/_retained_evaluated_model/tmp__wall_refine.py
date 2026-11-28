exec(open('/tmp/wall_compare.py').read().split('for x,b in itertools.product')[0].replace("print('aux mean/std'", "#print('aux mean/std'").replace("dif.argmin(1),np.round(dif.min(1),3))", "dif.argmin(1),np.round(dif.min(1),3))"))
from scipy.linalg import null_space
from scipy.optimize import minimize
N=null_space(np.ones((1,m))); w0=np.full(m,s['total_budget']/m); cap=s['u_box']; row=np.arange(y.size); ns=len(s['sensor_ij']); Z0=Z
old_axis=axis
def axis(k,v,h,r,mode):
 if isinstance(mode,str): return old_axis(k,v,h,r,mode)
 A=old_axis(k,v,h,r,'periodic').tolil()
 for i,j,bc in ((0,1,mode[0]),(len(k)-1,len(k)-2,mode[1])):
  if bc!='p':
   A[i,:]=0.; A[i,j]=-k[i]/h**2; A[i,i]=-A[i,j]+(r/h if bc=='r' else 0.)
 return A.tocsc()
def predict(S,mask):
 z=Z[mask]; A=S[mask]; yy=(y-S@w0)[mask]; B=np.column_stack((A@N,z)); theta=np.linalg.lstsq(B,yy,rcond=None)[0]; w=w0+N@theta[:m-1]
 if w.min() < -1e-9 or w.max() > cap+1e-9:
  P=lambda b:b-z@np.linalg.lstsq(z,b,rcond=None)[0]
  w,_=core._qp(P(A),P(y[mask]),1e-10,cap,s['total_budget']); gam=np.linalg.lstsq(z,y[mask]-A@w,rcond=None)[0]
 else: gam=theta[m-1:]
 return y-S@w-Z@gam,w
out=[]
for a,b,c,d in itertools.product('spr',repeat=4):
 x,wy=a+b,c+d
 S=forward(tuple(x),tuple(wy)); res,w=predict(S,row>=0)
 out.append((np.sqrt(np.mean(res**2)),x,wy,S,w))
for rm,x,wy,S,w in sorted(out,key=lambda q:q[0])[:12]:
 groups=row//4%(ns//4); cv=[predict(S,groups!=j)[0][groups==j] for j in np.unique(groups)]; shifts=row//ns; scv=[predict(S,shifts!=j)[0][shifts==j] for j in np.unique(shifts)]; r=predict(S,row>=0)[0]
 print(x,wy,'train',rm,'rosette pooled/worst',np.sqrt(np.mean(np.square(cv))),max(np.sqrt(np.mean(q*q)) for q in cv),'shift',np.round([np.sqrt(np.mean(q*q)) for q in scv],6),'max',np.max(abs(r)),'weights',np.round(w,4))

import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; b=c['single_bank_sweeps']; u,v,j=b['ref_signal'],b['raw_readout'],b['bank_id']; N=len(x)
def mask(y):
 m=np.median(y,axis=0); return np.max(abs(y-m)/np.maximum(np.median(abs(y-m),axis=0),1e-9),axis=1)<9
selfs={'tanh':([],lambda z,a:[np.tanh(a*z)]),'linear':([],lambda z,a:[np.tanh(a*z),z]),'cubic':([],lambda z,a:[np.tanh(a*z),z**3]),'quintic':([],lambda z,a:[np.tanh(a*z),z**5]),'septic':([],lambda z,a:[np.tanh(a*z),z**7]),'boundedcub':([(0,4,.6)],lambda z,a,d:[np.tanh(a*z),z**3/(1+d*z*z)]),'dualodd':([],lambda z,a:[np.tanh(a*z),z,z**3]),'tanh5+even':([],lambda z,a:[np.tanh(a*z),z**5,z*z]),'sinh':([(0.1,6,2)],lambda z,a,d:[np.tanh(a*z),np.sinh(d*z)-d*z])}
cross={'poly':([],lambda p:[p,p*p]),'rat':([(0,4,.4)],lambda p,d:[p,p*p/(1+d*p*p)]),'tanh':([(0.05,3,.6)],lambda p,d:[p,np.tanh(d*p*p)]),'squareclip':([(0.5,2,1.3)],lambda p,d:[p,np.minimum(p*p,d*d)]),'quartic':([],lambda p:[p,p*p,p**4]),'rationalboth':([(0,2,.15),(0,3,.3)],lambda p,d,f:[p/(1+d*p*p),p*p/(1+f*p*p)])}
for sn in selfs:
 for cn in cross:
  if sn not in ('tanh','quintic','dualodd') and cn not in ('poly','rat','quartic'): continue
  pars=[(.25,2.8,1.25)]+selfs[sn][0]+cross[cn][0]; lo,hi,start=np.array(pars).T
  e=np.zeros_like(y); eb=np.zeros_like(v); inv=np.zeros_like(y)
  for k in range(7):
   te=np.arange(N)%6==k if k<6 else np.zeros(N,bool); be=np.arange(len(u))%6==k if k<6 else np.zeros(len(u),bool); tr=~te; tr[tr]=mask(y[tr]); details=[]; ds=[]
   for h in range(4):
    jj=j==h; U=u[jj]; V=v[jj]; bt=~be[jj]; bt[bt]=mask(V[bt,None]); ds.append((U,V,bt,jj))
   def basis(z,p,par):
    ns=1+len(selfs[sn][0]); return np.column_stack([np.ones_like(z),*selfs[sn][1](z,*par[:ns]),*cross[cn][1](p,*par[ns:])])
   def fit(par,write=False):
    resid=[]
    for h,(U,V,bt,jj) in enumerate(ds):
     z=x[:,h]; p=x[:,(h+2)%4]; A=basis(z,p,par); B=basis(U,np.zeros_like(U),par); D=np.r_[A[tr],B[bt]]; Y=np.r_[y[tr,h],V[bt]]; C=np.linalg.lstsq(D,Y,rcond=None)[0]; resid.extend(D@C-Y)
     if write:
      e[te,h]=A[te]@C-y[te,h]; eb[jj&be]=B[be[jj]]@C-V[be[jj]]; details.append(C)
    return np.array(resid)
   out=least_squares(fit,start,bounds=(lo,hi),max_nfev=80); fit(out.x,True)
  good=mask(y); central=np.max(abs(x),axis=1)<.75; err=np.linalg.norm(e,axis=1); print(sn,cn,'par',np.round(out.x,3),'sweep',np.round(np.quantile(err[good],[.5,.95,1]),4),'central',np.round([np.sum(good&central),np.max(err[good&central])],4),'bank',np.round(np.quantile(abs(eb[abs(v)<4]),[.95,1]),4),'coeff',np.round(details,3).tolist() if cn in ('rat','quartic') and sn in ('tanh','quintic','dualodd') else '',flush=True)

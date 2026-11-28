import pickle,numpy as np,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; I=np.eye(4); pairs=list(itertools.combinations(range(4),2)); B=np.linalg.lstsq(np.c_[np.ones(24),s['panel_b']['u']],s['panel_b']['v'],rcond=None)[0];print('B sv',np.linalg.svd(B[1:])[1]); L=np.column_stack([I]+[I[:,i:i+1]+sgn*I[:,j:j+1] for i,j in pairs for sgn in [1,-1]]); P=lambda u:np.column_stack([u[:,i]*u[:,j] for i,j in pairs]); A=lambda u:np.c_[np.ones(len(u)),u]; fs={'square':lambda x,k:x*x,'cross':None,'cube':lambda x,k:x**3,'sin':lambda x,k:np.sin(k*x),'cos':lambda x,k:np.cos(k*x),'tanh':lambda x,k:np.tanh(k*x),'rational':lambda x,k:x/(1+k*x*x),'softabs':lambda x,k:x/(1+k*abs(x)),'ratsq':lambda x,k:x*x/(1+k*x*x),'abs':lambda x,k:abs(x)}
for key,out in [('panel_a','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[key]['u'];v=s[key][out];a=A(u);mask=np.ones(len(u),bool)
 for _ in range(7):
  e=np.linalg.norm(a@np.linalg.lstsq(a[mask],v[mask],rcond=None)[0]-v,axis=1)
  if key=='panel_a':mask=e<=max(.42,3*np.median(e[mask]))
 print(key,out,'mask',mask.sum(),'base',np.sqrt(np.mean(e[mask]**2)/4));w=[]
 for name,f in fs.items():
  for coord,z in [('raw',u),('rot',u@B[1:]),('shiftrot',u@B[1:]+B[0]),('revrot',u@B[1:].T),('paired',u@L)]:
   if name=='cross' and coord=='paired':continue
   def fit(k):
    h=P(z) if name=='cross' else f(z,k);D=np.c_[a,h];d=D[mask];y=v[mask];coef=np.linalg.lstsq(d,y,rcond=1e-10)[0];err=d@coef-y;hat=(d*np.linalg.pinv(d).T).sum(1);return np.mean(err**2)**.5,np.mean((err/np.maximum(1-hat[:,None],1e-8))**2)**.5
   if name in ['square','cross','cube','abs']:k=1.
   else:
    grid=np.r_[np.linspace(.08,3.5,25),.16,.22,1.25,2.15];k=min(grid,key=lambda k:fit(k)[0]);k=minimize_scalar(lambda k:fit(k)[0],bounds=(max(.001,k-.15),k+.15),method='bounded').x
   err,loo=fit(k);w.append((loo,err,name,coord,k))
 for row in sorted(w)[:16]:print('%10.6g %10.6g %-10s %-10s %.7f'%row)
 print('u head',u[:3]);print('v head',v[:3])

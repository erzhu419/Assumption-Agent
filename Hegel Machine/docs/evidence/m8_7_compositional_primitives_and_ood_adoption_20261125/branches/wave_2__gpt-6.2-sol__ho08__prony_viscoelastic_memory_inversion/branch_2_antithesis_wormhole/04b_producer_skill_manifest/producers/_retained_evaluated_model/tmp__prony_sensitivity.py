import pickle,numpy as np
from scipy.optimize import nnls, least_squares
from prony_probe import basis
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); x=p['experiment']; t=x['train_t']; e=x['train_strain']; y=x['train_stress_raw']; d=np.column_stack((np.ones_like(t),t/30-.5)); tt=x['test_t']; et=x['test_strain']; gt=p['gt_test_stress']; grid=np.geomspace(.02,20,100); a=basis(t,e,grid,'ramp'); at=basis(tt,et,grid,'ramp'); N=np.linalg.norm(gt)

def report(label,cc,bb,taus,train,test):
 r=y-train; s=1.4826*np.median(abs(r-np.median(r))); print(label,'G %.4f test %.5f MAD %.5f cleanRMS %.5f'%(cc[0],np.linalg.norm(test-gt)/N,s,np.sqrt(np.mean(np.sort(r*r)[:len(y)-12]))),'drift',bb.round(4),'tau',taus.round(4),'Gm',cc[1:].round(3),'flags',sum(abs(r)>6*s)); return r,s

def gridfit(f,mask):
 w=mask.astype(float)
 for k in range(40):
  aw=a*w[:,None]; dw=d*w[:,None]; yw=y*w; q,_=np.linalg.qr(dw,mode='reduced'); c=nnls(aw-q@(q.T@aw),yw-q@(q.T@yw),maxiter=3000)[0]; b=np.linalg.lstsq(dw,yw-aw@c,rcond=None)[0]; r=y-a@c-d@b; s=1.4826*np.median(abs(r[mask]-np.median(r[mask]))); u=abs(r)/(s*abs(f)); nw=mask*((1+u*u)**(-.5) if f>0 else (u<1));
  if np.max(abs(w-nw))<1e-5: break
  w=nw
 return c,b,r,s
mask=np.ones(len(t),bool)
for f in [1,2,3,5,8]:
 c,b,r,s=gridfit(f,mask); keep=c[1:]>1e-6; report('grid cauchy '+str(f),np.r_[c[0],c[1:][keep]],b,grid[keep],a@c+d@b,at@c)
c,b,r,s=gridfit(3,mask)
for i in np.argsort(abs(r))[-20:][::-1]: print('residual',t[i],round(r[i],4))
for i in range(0,len(t)-1,60):
 ix=np.arange(i,i+60); rr=r[ix]; rk=rr[abs(rr)<5*s]; print('time slice',t[i],t[i+59],e[ix].min(),e[ix].max(),round(np.mean(rk),5),round(np.std(rk),5))
for m in [2,3,4,5,6]:
 def predict(v,T,E): return basis(T,E,np.exp(v[m+3:]),'ramp')@v[:m+1]
 def fun(v): return (predict(v,t,e)+d@v[m+1:m+3]-y)[mask]
 v0=np.r_[np.r_[12.,np.full(m,12.)],b,np.linspace(np.log(.045),np.log(10),m)]; lb=np.r_[np.zeros(m+1),[-np.inf]*2,np.full(m,np.log(.02))]; ub=np.r_[[np.inf]*(m+3),np.full(m,np.log(20))];
 for f in [1.5,3,5,0]:
  if f==0: mask=abs(r)<6*s
  else: mask=np.ones(len(t),bool)
  v=least_squares(fun,v0,bounds=(lb,ub),loss='cauchy' if f else 'linear',f_scale=max(f*s,1e-8),max_nfev=180,ftol=1e-10).x
  report(f'm{m} f{f}',v[:m+1],v[m+1:m+3],np.exp(v[m+3:]),predict(v,t,e)+d@v[m+1:m+3],predict(v,tt,et))
cg=nnls(at[:,1:],gt-p['gt_G_inf']*at[:,0],maxiter=3000)[0]; print('public clean representability witness',list(zip(grid[cg>1e-4].round(4),cg[cg>1e-4].round(3))))

from bridge_probe import D,X,Y,N
import numpy as np
from scipy.optimize import least_squares
S=D['calibration_load_cells']['single_channel_sweeps']; lab=S['lab_id']; sx=S['true_stress']; sy=S['cmd_stress']; P=[2,3,0,1]
clean=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<8; sc=abs(sy-np.median(sy))<8
forms={'cos':lambda v,k:1-np.cos(k*v),'rat':lambda v,k:v*v/(1+k*v*v),'exp':lambda v,k:-np.expm1(-k*v*v),'poly':lambda v,k:v**4}
def design(u,v,z,g,h):
 a=[np.ones(len(u)),u,np.sin(z[0]*u),v,forms[g](v,z[1])]
 if g=='poly': a.append(v*v)
 if h=='cubic': a.append(u**3)
 if h=='sin3': a.append(np.sin(3*z[0]*u))
 if h=='inter': a.extend([u*v,u*u*v,u*v*v])
 return np.array(a).T
def model(m,g,h,robust=False,sm=sc):
 pred=[]; co=[]; swept=[]
 for j,p in enumerate(P):
  s=sm&(lab==j); u=np.r_[X[m,j],sx[s]]; v=np.r_[X[m,p],np.zeros(s.sum())]; y=np.r_[Y[m,j],sy[s]]; w=np.ones(len(y))
  def coeff(z):
   a=design(u,v,z,g,h); b=np.linalg.lstsq(a*w[:,None],y*w,rcond=None)[0];return a@b-y,b
  start=[2.5,1.6 if g=='cos' else .3]
  for _ in range(6 if robust else 1):
   z=least_squares(lambda z:coeff(z)[0]*w,start,bounds=([1,.05],[4,3.5]),max_nfev=75).x; e,b=coeff(z); start=z; w=(1+(e/max(.00025,2.5*np.median(abs(e))))**2)**(-.5)
  pred.append(design(X[:,j],X[:,p],z,g,h)@b); swept.extend(zip(np.where(s)[0],e[m.sum():])); co.append((z,b))
 return np.array(pred).T,co,swept
if __name__=='__main__':
 folds=np.random.default_rng(8).permutation(N)%5; edge=np.max(abs(X),axis=1)>=np.quantile(np.max(abs(X),axis=1),.8)
 for g in forms:
  for h in ['none','cubic','sin3','inter']:
   pred,co,se=model(clean,g,h,True); e=np.linalg.norm(pred-Y,axis=1); err=np.zeros(N)
   for f in range(5):
    q=folds==f; pp,_,_=model(clean&~q,g,h,True); err[q]=np.linalg.norm((pp-Y)[q],axis=1)
   pp,_,_=model(clean&~edge,g,h,True); ee=np.linalg.norm(pp-Y,axis=1);print(g,h,'fit/OOF/edge p90,max',*[np.round(np.quantile(q,[.9,1]),7).tolist() for q in [e[clean],err[clean],ee[clean&edge]]]);
   if h=='none':
    print('params',[(np.round(z,5).tolist(),np.round(b,6).tolist()) for z,b in co]); idx=np.where(clean)[0][np.argsort(e[clean])[-4:]]; print('worst rows',idx.tolist(),np.round((pred-Y)[idx],6).tolist(),'worst sweep',sorted(se,key=lambda q:abs(q[1]))[-2:]);
 for split in ['train','test']:
  for k in ['past_channel_raw','hold_channel_raw']:
   a=np.asarray(D[split][k]).reshape(-1,4); print(split,k,len(a),a.min(0),a.max(0))

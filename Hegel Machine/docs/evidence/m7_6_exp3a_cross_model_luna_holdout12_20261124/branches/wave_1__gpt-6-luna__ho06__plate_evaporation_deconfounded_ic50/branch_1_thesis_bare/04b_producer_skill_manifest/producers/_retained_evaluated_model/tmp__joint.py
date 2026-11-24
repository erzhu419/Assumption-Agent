import pickle, numpy as np
from scipy.optimize import least_squares
x=pickle.load(open('/holdout_task/public_screen.pkl','rb'))
plates=x['plates']; ids=sorted({w['compound_id'] for p in plates for w in p['wells'] if w['well_type']=='COMPOUND'}); ci={k:i for i,k in enumerate(ids)}
R=[]; Y=[]; typ=[]; cidx=[]; dose=[]; pidx=[]
for pi,p in enumerate(plates):
 for w in p['wells']:
  r=w['row']/15.; c=w['col']/23.
  R.append([1,4*r*(1-r),4*c*(1-c),(r-.5)*(c-.5),(r-.5)**2,(c-.5)**2])
  Y.append(w['fluorescence']); typ.append(w['well_type']); pidx.append(pi)
  if w['well_type']=='COMPOUND': cidx.append(ci[w['compound_id']]);dose.append(w['log10_conc_uM'])
  else:cidx.append(-1);dose.append(0.)
R=np.array(R);Y=np.array(Y);typ=np.array(typ);pidx=np.array(pidx);cidx=np.array(cidx);dose=np.array(dose)
npl=len(plates); n=npl*7+len(ids)*2
q0=np.zeros(n)
for j,p in enumerate(plates):
 kill=[w['fluorescence'] for w in p['wells'] if w['well_type']=='POS_KILL']; dm=[w['fluorescence'] for w in p['wells'] if w['well_type']=='DMSO']
 q0[7*j]=np.median(kill);q0[7*j+1]=np.log(np.median(dm)-np.median(kill))
q0[npl*7::2]=np.linspace(-2,1.7,len(ids));q0[npl*7+1::2]=1
ctrl=np.where(typ=='DMSO',1.,np.where(typ=='POS_KILL',0.,np.nan))
def pred(q):
 out=np.empty(len(Y))
 for j in range(npl):
  ix=pidx==j; b=q[7*j]; amp=np.exp(q[7*j+1]); g=np.exp(np.clip(R[ix,1:]@q[7*j+2:7*j+7],-.5,.5))
  v=ctrl[ix].copy(); cc=cidx[ix]>=0
  if cc.any():
   h=q[npl*7+2*cidx[ix][cc]]; s=q[npl*7+2*cidx[ix][cc]+1]
   v[cc]=1/(1+np.exp(np.clip(s*(dose[ix][cc]-h)*np.log(10),-35,35)))
  out[ix]=b+amp*g*v
 return out
scale=np.maximum(np.abs(Y)*.015,8)
lb=np.r_[np.tile([0,np.log(100),-1,-1,-1,-1,-1],npl),np.tile([-4,.3],len(ids))]; ub=np.r_[np.tile([1000,np.log(10000),1,1,1,1,1],npl),np.tile([3,3],len(ids))]
fit=least_squares(lambda q:(pred(q)-Y)/scale,q0,loss='soft_l1',f_scale=1.,max_nfev=3000,bounds=(lb,ub))
res={k:fit.x[npl*7+2*i:npl*7+2*i+2] for k,i in ci.items()}; print({k:np.round(v,3).tolist() for k,v in res.items()});print('err',max(abs(res[k][0]-x['gt'][k]['log10_ic50_uM']) for k in ids),max(abs(res[k][1]-x['gt'][k]['hill_slope']) for k in ids));print('cost',fit.cost)

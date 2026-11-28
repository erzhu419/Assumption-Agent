import pickle,itertools,numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; X=c['true_channel_loads']; Y=c['bridge_raw']; print('sweep fields',list(c['single_channel_sweeps'])); print('specimen ranges',[(p,k, np.round(D[p][k].reshape(-1,4).min(0),2).tolist(),np.round(D[p][k].reshape(-1,4).max(0),2).tolist()) for p in ['train','test'] for k in ['past_channel_raw','hold_channel_raw']]); terms=[s for k in range(4) for s in itertools.combinations_with_replacement(range(4),k)]
for j in range(4):
 for model in ('archive','own_poly','all_cross','full3'):
  t=[s for s in terms if len(s)<=1 or (model=='full3') or s==(j,j,j) or s==tuple(sorted((j,(j+1)%4))) or (model in ('own_poly','all_cross') and s==(j,j)) or (model=='all_cross' and len(s)==2)]
  a=np.column_stack([np.prod(X[:,s],axis=1) for s in t]); y=Y[:,j]; p=least_squares(lambda b:a@b-y,np.linalg.lstsq(a,y,rcond=None)[0],jac=lambda b:a.copy(),loss='soft_l1',f_scale=.003).x; e=a@p-y; clean=abs(e)<1; p=np.linalg.lstsq(a[clean],y[clean],rcond=None)[0]; e=(a@p-y)[clean]; print(j,model,clean.sum(),np.round(np.quantile(abs(e),[.5,.9,1]),6),[(s,round(v,5)) for s,v in zip(t,p) if len(s)>1 and abs(v)>.008])

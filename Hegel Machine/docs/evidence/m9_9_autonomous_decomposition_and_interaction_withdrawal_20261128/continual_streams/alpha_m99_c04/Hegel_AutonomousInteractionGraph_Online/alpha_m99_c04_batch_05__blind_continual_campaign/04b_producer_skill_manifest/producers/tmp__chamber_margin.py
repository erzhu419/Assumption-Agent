from chamber_probe import *
import downstream_solver as m
import json
for a in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 A=np.array(a['facility_telemetry_trace']); print(a['completed_campaign_short_name'], 'mean', A.mean(0).round(3), 'tail', A[-4:].mean(0).round(3), 'std', A.std(0).round(3))
xd,_=m._features(g,g)
obs=np.c_[g['strain_curves_obs'],R]
lab=g['lab_id']; complete=np.isfinite(g['coupon_features_raw']).all(1); stratum=g['stratum']
for excluded in [-1,*np.unique(tr['lab_id'])]:
 sel=tr['lab_id']!=excluded; sub={k:v[sel] for k,v in tr.items()}; a,b=m._fit(sub,x[sel],dt,t)
 q,r=m._chamber_response(g,xd,t,a,b); e=obs-np.c_[q,r]
 age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]; z=P@(-np.expm1(-dt[:,None]/a)*np.exp(-age[:,None]/a))
 k=np.r_[-a[None,:]*np.expm1(-t[:,None]/a),[[1.,1.]]]; dk=.2*k[:,0]+.55*k[:,1]+.25*np.r_[t,1.]
 j=-z[:,1,None]*(b[1]/a[1])*k[None,:,1]
 proj=lambda v:v-((v@dk)/(dk@dk))[:,None]*dk
 for name in ('baseline','slow history raw','slow history drift-orthogonal'):
  pred=e.copy(); coefficients=[]
  for l in np.unique(lab):
   learn=(lab!=l)&complete; test=lab==l
   J,Q=(j,e) if name.endswith('raw') else (proj(j),proj(e))
   alpha=0. if name=='baseline' else np.sum(J[learn]*Q[learn])/np.sum(J[learn]**2)
   coefficients.append(alpha); pred[test]-=alpha*j[test]
  masks=[lab==l for l in np.unique(lab)]+[complete,~complete]+[stratum==s for s in np.unique(stratum)]
  v=np.array([[abs(pred[f,:-1]).mean(),abs(pred[f,-2]).mean(),abs(pred[f,-1]).mean()] for f in masks]);
  print('benchtop withheld',excluded,name,'worst',v.max(0).round(5),'complete',v[4].round(5),'coeff',np.round(coefficients,4))

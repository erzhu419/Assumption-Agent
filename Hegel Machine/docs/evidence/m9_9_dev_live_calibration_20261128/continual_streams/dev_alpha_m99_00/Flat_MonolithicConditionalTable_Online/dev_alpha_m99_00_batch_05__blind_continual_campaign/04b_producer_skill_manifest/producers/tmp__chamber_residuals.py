from chamber_compare import *
np.set_printoptions(precision=4,suppress=True,linewidth=180)
kh=k@th[:2]; rh=np.sum(th[:2]/tau)
Y=np.column_stack((r-hold*rh,y-hold[:,None]*kh)); T=np.r_[0,t]
xa=m._features(obs)[good]; xb=m._features(tr,te,obs)[n+len(te['lab_id']):][good]
for label,xx in [('train/test ref',x),('diag ref',xa),('all ref',xb)]:
 sc=th[2]+xx@th[3:]; print('\n',label,'score',sc)
 print('kernel ratio median',np.median(Y/sc[:,None],0)); print('score/rate residual',np.round(Y[:,0]-sc,4))
 print('linear kernel coefficient',np.linalg.lstsq(np.c_[np.ones(len(sc)),sc],Y,rcond=None)[0]); print('rank1 rms',np.sqrt(np.mean((Y-sc[:,None]*(sc@Y/(sc@sc)))**2,0)))
print('lab,hold,history,gaps,score and scaled residuals')
print(np.c_[o['lab_id'],hold,past,score,Y[:,[0,1,4,8,12]]]); print('Y singular values',np.linalg.svd(Y,compute_uv=False)); print('reference center,scale by group')
for l in np.unique(lab):
 print(l,[(np.median(ref.mean(0)),np.median(ref.std(0))) for ref in [tr['ref_coupons'][tr['lab_id']==l],te['ref_coupons'][te['lab_id']==l],refs[lab==l],obs['ref_coupons'][ol==l]]])
for label,v in [('residual rate',Y[:,0]-score),('residual final',Y[:,-1]-score*(-np.expm1(-t[-1])))]:
 cols=np.c_[x,hold,past,hold*score,score**2,score**3]
 print(label,'correlations with [features,hold,history,hold*score,score**2,score**3]',np.corrcoef(np.c_[v,cols].T)[0,1:])

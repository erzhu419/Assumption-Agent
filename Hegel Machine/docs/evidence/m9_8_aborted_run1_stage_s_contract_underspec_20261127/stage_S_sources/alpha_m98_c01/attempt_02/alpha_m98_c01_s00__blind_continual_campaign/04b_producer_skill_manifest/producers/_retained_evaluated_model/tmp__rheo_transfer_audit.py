exec(open('/tmp/rheo_margin.py').read().split("models=")[0])
for name,v in [('train',tr),('chamber',ch),('test',te)]:
 print(name,'fields and shapes',[(k,np.shape(a)) for k,a in v.items()]); print('lab sample counts',np.unique(v['lab_id'],return_counts=True));
 if 'aux_chamber_telemetry' in v:
  u=v['aux_chamber_telemetry']; print('telemetry mean and spread by lab',[(int(k),np.round(np.mean(u[v['lab_id']==k],axis=0),3).tolist(),np.round(np.std(u[v['lab_id']==k],axis=0),3).tolist()) for k in np.unique(v['lab_id'])]);
print('pilot lab/stratum/score/slope',np.c_[g,ch['stratum'],score,y].round(4).tolist());
for l in L:
 j=g==l; xx,_=s._standardize(te,ref,fill); testscore=xx[te['lab_id']==l,1:]@b[3:]; print('lab and score range',l,np.round([score[j].min(),score[j].max(),testscore.min(),testscore.max()],3));

def measure(F,pen,groups):
 er=E.copy()
 for k in np.unique(groups):
  j=groups==k; er[j]-=ridge(F,~j,j,pen)[:,None]*z
 by=np.array([[np.mean(abs(er[g==l,:-1])),np.mean(abs(er[g==l,-1]))] for l in L]);return np.r_[[np.mean(abs(er[:,:-1])),np.mean(abs(er[:,-1]))],by.max(axis=0)].round(4).tolist()
folds=(np.arange(len(g)),ch['stratum'],g)
for name,F,p in [('base',local,.08),('gain-prior',local,np.r_[np.full(len(L),.08),np.full(len(L),.16)]),('shared gain only',np.c_[local,score],np.r_[np.full(2*len(L),.08),.16]),('shared offset only',np.c_[local,np.ones(len(g))],np.r_[np.full(2*len(L),.08),.16]),('lab-stratum',np.c_[local,np.equal.outer(ch['stratum'],np.unique(ch['stratum']))],np.r_[np.full(2*len(L),.08),np.full(len(np.unique(ch['stratum'])),.35)])]:
 print(name,[measure(F,p,f) for f in folds]);

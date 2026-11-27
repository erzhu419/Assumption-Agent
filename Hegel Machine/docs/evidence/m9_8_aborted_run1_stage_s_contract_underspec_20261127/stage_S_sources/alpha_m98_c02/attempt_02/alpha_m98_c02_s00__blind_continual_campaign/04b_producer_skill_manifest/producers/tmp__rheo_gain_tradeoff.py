exec(open('/tmp/rheo_labgain.py').read().split("for mode in ['base','centered_gain']:")[0])
lab=tr['lab_id']; clab=ch['lab_id']; subsets=[pb[-1]]+[pb[-1]&(lab!=l) for l in np.unique(lab)]
pairs=[(fit(s,'base'),fit(s,'centered_gain')) for s in subsets]
for mix in (0., .25, .5, .75, 1.):
 def combined(z):
  (t0,w0),(t1,w1)=z; return np.exp((1-mix)*np.log(t0)+mix*np.log(t1)),(1-mix)*w0+mix*w1
 e,r=errors(ch,pc,combined(pairs[0])); co=report(e,r,pc[-1]); mi=report(e,r,~pc[-1]); worst=np.max([report(e,r,clab==l) for l in np.unique(clab)],axis=0)
 folds=[]; delta=[]
 for l,z in zip(np.unique(lab),pairs[1:]):
  e,r=errors(tr,pb,combined(z)); folds.append(report(e,r,(lab==l)&pb[-1])); delta.append(combined(z)[0]);
 print(mix,'chamber complete',co,'missing',mi,'worst lab',worst,'bench worst complete',np.max(folds,axis=0),'slow tau fold range',np.min(delta,axis=0)[1],np.max(delta,axis=0)[1])

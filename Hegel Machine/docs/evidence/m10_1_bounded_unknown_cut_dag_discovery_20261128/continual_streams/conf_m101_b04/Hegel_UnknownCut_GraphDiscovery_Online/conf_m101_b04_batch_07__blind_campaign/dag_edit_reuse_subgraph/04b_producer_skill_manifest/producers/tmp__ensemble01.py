exec(open('/tmp/link01.py').read().split('print(\'linear univariate')[0] + open('/tmp/link01.py').read().split('def evaluate',1)[1].split('cases=list',1)[0].join(['def evaluate','']))
idx=np.arange(N)
# Average smooth link families instead of selecting a winner on noisy inner splits.
for label,cases in [('link',list(itertools.product((1,2),(1,2),(0,),(.01,.03,.08)))),('link_amplitude',list(itertools.product((1,2),(1,2),(0,1,2),(.01,.03,.08))))]:
 for te in folds:
  tr=np.setdiff1d(idx,te); e=np.mean([evaluate(tr,te,c) for c in cases],axis=0); a=mod._scalar(b[tr],b[te],y[tr]); print(label,len(te),*[round(np.mean(abs(p-y[te])),6) for p in (a,e,.75*e+.25*a)])

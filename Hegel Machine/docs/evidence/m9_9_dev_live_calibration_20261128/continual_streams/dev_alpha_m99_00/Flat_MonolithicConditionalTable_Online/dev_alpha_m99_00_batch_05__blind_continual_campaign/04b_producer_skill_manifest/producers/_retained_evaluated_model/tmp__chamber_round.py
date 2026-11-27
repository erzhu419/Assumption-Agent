from chamber_kernel_cv import *
for p in ((.25,.6,7.5),(.25,.6,7.0),(.25,.65,7.5)):
 e=score[:,None]*kernel('two-exp',p,t)-Y
 print('rounded kernel',p,'worst lab/early/late',*[max(abs(e[labs==l,sl]).mean() for l in np.unique(labs)) for sl in (slice(None),slice(0,4),slice(-4,None))])
for lab0 in np.unique(labs):
 print('fold',lab0,'two-exp',fit('two-exp',labs!=lab0))

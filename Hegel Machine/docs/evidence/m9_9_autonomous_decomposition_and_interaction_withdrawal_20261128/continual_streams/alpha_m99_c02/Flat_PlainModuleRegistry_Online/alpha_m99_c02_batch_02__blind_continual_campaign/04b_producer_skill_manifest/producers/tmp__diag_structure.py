import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from chamber_probe import *
np.set_printoptions(precision=4,suppress=True,linewidth=150)
base=c@theta; dr=r[:,2:]@theta[2:]; gaps=r[:,:2]*taus
print('columns: lab, hold, drift, gaps, actual-minus-baseline at early, middle and late times, initial-rate difference')
print(np.c_[di['lab_id'],h,dr,gaps,(y-base)[:,[0,3,7,11]],q-r@theta])
print('center scale train/test, train/diagnostics, diagnostics only:')
for lab in np.unique(tr['lab_id']):
 print(lab,[(round(np.median(z.mean(0)),4),round(np.median(z.std(0)),4)) for z in [np.r_[tr['ref_coupons'][tr['lab_id']==lab],s['ref_coupons'][s['lab_id']==lab]] for s in (te,di)]+[di['ref_coupons'][di['lab_id']==lab]]])
A=np.c_[q-r@theta,y-base]; print('correction singular values uncentered and centered:',np.linalg.svd(A,compute_uv=False),np.linalg.svd(A-A.mean(0),compute_uv=False)); print('diagnostic curve median total variation of second differences',np.median(abs(np.diff(y-base,n=2,axis=1))))

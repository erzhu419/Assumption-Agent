import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_beta_m98_00_batch_02_solver as base
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=item['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']
def matrix(x,d):
    return np.column_stack([np.ones(len(x))]+[np.prod(x[:,ids],axis=1) for k in range(1,d+1) for ids in itertools.combinations_with_replacement(range(4),k)])
np.set_printoptions(precision=5,suppress=True)
for d in (1,2,3):
    a=matrix(x,d); errs=[]
    for idx in np.array_split(np.arange(len(x)),5):
        sel=np.setdiff1d(np.arange(len(x)),idx); fit=np.linalg.lstsq(a[sel],y[sel],rcond=None)[0]; errs.append(np.sqrt(np.mean((a[idx]@fit-y[idx])**2,axis=0)))
    print('calibration degree',d,'worst-fold RMS by channel',np.max(errs,axis=0),'full RMS',np.sqrt(np.mean((a@np.linalg.lstsq(a,y,rcond=None)[0]-y)**2,axis=0)))
a=matrix(x,1); fit=np.linalg.lstsq(a,y,rcond=None)[0]; print('affine',fit,'ref range',x.min(0),x.max(0))
b=c['single_bank_sweeps']; print('single bank',[(k,np.shape(v),np.asarray(v)[:8]) for k,v in b.items()]); bank=np.asarray(b['bank_id']); sig=np.asarray(b['ref_signal']); raw=np.asarray(b['raw_readout']);
for i in np.unique(bank):
    mask=bank==i; print('single sweep bank',i,'range',sig[mask].min(),sig[mask].max(),'affine/quad/cubic RMS',[np.sqrt(np.mean((raw[mask]-np.polynomial.polynomial.polyval(sig[mask],np.polynomial.polynomial.polyfit(sig[mask],raw[mask],d)))**2)) for d in (1,2,3)])
for fx,fy in itertools.product((0.,1.),repeat=2):
    st=dict(s,robin_x=fx*s['robin_x'],robin_y=fy*s['robin_y']); S,_=base._design(st)
    for j,p in enumerate(item['campaign_diagnostics']):
        z=(S@p['w_pulse']).reshape(-1,4); target=p['y_pulse_raw'].reshape(-1,4)
        print('pulse',j,'wall',fx,fy,'design range',z.min(),z.max(),'affine RMS by channel',np.sqrt(np.mean((matrix(z,1)@fit-target)**2,axis=0)),'degree-2 RMS',np.sqrt(np.mean((matrix(z,2)@np.linalg.lstsq(matrix(x,2),y,rcond=None)[0]-target)**2)))

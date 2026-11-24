import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from mag_calibrator import calibrate_and_estimate_heading


def metrics(session, soft, bias, yaw):
    out = calibrate_and_estimate_heading(session)
    S, b, h = (out[k] for k in ('S_soft', 'b_hard', 'heading_deg'))
    assert S.shape == (3,3) and b.shape == (3,) and h.shape == yaw.shape
    assert np.all(np.isfinite(np.r_[S.ravel(),b,h]))
    assert np.max(abs(S-S.T)) < 1e-12 and np.linalg.eigvalsh(S)[0] > 0
    assert abs(np.linalg.det(S)-1) < 1e-10 and np.all((-180 < h) & (h <= 180))
    err = (h - yaw + 180) % 360 - 180
    idx = np.clip(np.floor((yaw+180)/90).astype(int), 0, 3)
    return np.array([np.sqrt(np.mean(err**2)), max(np.sqrt(np.mean(err[idx==i]**2)) for i in np.unique(idx)), np.linalg.norm(b-bias), np.linalg.norm(S-soft)])


def report(name, rows):
    a = np.stack(rows)
    print(name, 'cases',len(rows),'max rmse/quadrant/bias/soft:',np.round(a.max(axis=0),6), 'rmse/bias failures:', ((a[:,0]>.45)|(a[:,2]>.35)).sum(), 'worst heading/bias case:',np.argmax(a[:,0]),np.argmax(a[:,2]))


with open('/holdout_task/public_mag.pkl','rb') as f:
    p = pickle.load(f)
s,g = p['session'],p['gt']; rp=s['calib_roll_pitch_rad']
def public(mask):
    return metrics(dict(s,calib_m_raw=s['calib_m_raw'][mask],calib_roll_pitch_rad=rp[mask]),g['S_soft'],g['b_hard'],g['gt_heading_deg'])
report('public all',[public(slice(None))])
report('public leave-one-tilt-cell-out',[public(~((np.digitize(rp[:,0],[-.25,.25])==i)&(np.digitize(rp[:,1],[-.22,.22])==j))) for i in range(3) for j in range(3)])
report('public calibrate on contiguous thirds',[public(ids) for ids in np.array_split(np.arange(len(rp)),3)])


def simulate(seed, mode):
    rng=np.random.default_rng(seed)
    N, extent, axes, field, noise, spike = {
        'nominal':(420,[.75,.65],[1.22,.96,1/1.1712],[31,41],.08,20),
        'short-narrow-tilt':(140,[.35,.28],[1.22,.96,1/1.1712],[29,42],.10,25),
        'steep-field-and-distortion':(420,[.75,.65],[1.65,1.03,1/1.6995],[18,49],.09,35),
        'clustered-yaw-and-spikes':(300,[.60,.50],[1.35,1.05,1/1.4175],[26,-41],.09,50),
    }[mode]
    U=Rotation.random(random_state=rng).as_matrix(); S=(U*np.array(axes))@U.T
    b=rng.uniform(-55,55,3); B=np.array([field[0],0,field[1]])
    def samples(rp, yaw):
        R=Rotation.from_euler('ZYX',np.column_stack((yaw,rp[:,1],rp[:,0])))
        return R.inv().apply(np.broadcast_to(B,(len(rp),3)))@S + b + rng.normal(0,noise,(len(rp),3))
    rp=rng.uniform(-1,1,(N,2))*extent; yaw=rng.uniform(-np.pi,np.pi,N)
    if mode=='clustered-yaw-and-spikes':
        yaw[:int(N*.7)] = rng.normal(.8,.28,int(N*.7))
    m=samples(rp,yaw); k=round(N*.04)
    ids=rng.choice(N,k,replace=False); direction=rng.normal(size=(1 if mode=='clustered-yaw-and-spikes' else k,3)); direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    m[ids]+=direction*rng.uniform(.3,1.7,(k,1))*spike
    tp=rng.uniform(-1,1,(160,2))*[.45,.38]
    ty=np.linspace(-np.pi,np.pi,160,endpoint=False)
    return metrics(dict(calib_m_raw=m,calib_roll_pitch_rad=rp,test_m_raw=samples(tp,ty),test_roll_pitch_rad=tp),S,b,np.degrees(ty))

for mode in ['nominal','short-narrow-tilt','steep-field-and-distortion','clustered-yaw-and-spikes']:
    report(mode,[simulate(seed,mode) for seed in range(24)])

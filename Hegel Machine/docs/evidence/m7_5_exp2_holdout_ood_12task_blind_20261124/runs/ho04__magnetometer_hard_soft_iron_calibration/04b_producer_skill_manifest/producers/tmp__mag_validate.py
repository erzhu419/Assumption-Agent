import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import mag_calibrator as m

p = pickle.load(open('/holdout_task/public_mag.pkl', 'rb'))
s, g = p['session'], p['gt']

def errors(result, truth):
    yaw = (result['heading_deg'] - truth['gt_heading_deg'] + 180) % 360 - 180
    return np.sqrt(np.mean(yaw**2)), np.linalg.norm(result['b_hard'] - truth['b_hard'])

def report(name, metrics):
    a = np.asarray(metrics)
    print(f'{name:27s} n={len(a):2d} median={np.median(a, axis=0).round(5)} max={a.max(0).round(5)}', flush=True)

def subset(mask):
    return {k: (v[mask] if k.startswith('calib_') else v) for k,v in s.items()}

if __name__ == '__main__':
    axes = m._level_axes(s['calib_roll_pitch_rad'])
    level = np.einsum('nij,nj->ni', axes, (s['calib_m_raw']-g['b_hard']) @ np.linalg.inv(g['S_soft']))
    sector = ((np.arctan2(-level[:,1], level[:,0])+np.pi)*8/(2*np.pi)).astype(int)%8
    rp = s['calib_roll_pitch_rad']
    quadrant = (rp[:,0]>0).astype(int) + 2*(rp[:,1]>0)
    block = np.arange(len(rp))//70
    for key,group in [('calibration yaw sectors',sector),('calibration tilt quadrants',quadrant),('calibration time blocks',block)]:
        report(key,[errors(m.calibrate_and_estimate_heading(subset(group!=i)),g) for i in np.unique(group)])

def synthetic(seed, n=420, tilt=1., h=23.5, soft=.20, spike='random'):
    rng = np.random.default_rng(seed)
    q,_ = np.linalg.qr(rng.normal(size=(3,3)))
    shape = (q * np.exp(np.array([-1.,-.25,1.25])*soft)) @ q.T
    bias = rng.uniform(-65,65,size=3)
    def sample(rp,yaw,noise):
        v = np.column_stack((h*np.cos(yaw),-h*np.sin(yaw),np.full(len(yaw),42.)))
        u = np.einsum('nij,ni->nj', m._level_axes(rp),v)
        return u @ shape + bias + rng.normal(0,noise,size=u.shape)
    rp = rng.uniform(-1,1,size=(n,2))*[.75*tilt,.65*tilt]
    yaw = rng.uniform(-np.pi,np.pi,size=n)
    x = sample(rp,yaw,.10)
    k = round(.04*n)
    ix = rng.choice(n,k,replace=False) if spike!='cluster' else np.argsort(yaw)[-k:]
    if spike=='random':
        x[ix] += rng.normal(0,12,size=(k,3))
    else:
        direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
        x[ix] += rng.uniform(7,30,size=(k,1))*direction
    r = rng.uniform(-1,1,size=(180,2))*[.4,.35]
    y = np.linspace(-np.pi,np.pi,180,endpoint=False)
    return dict(calib_m_raw=x,calib_roll_pitch_rad=rp,test_m_raw=sample(r,y,.06),test_roll_pitch_rad=r), dict(b_hard=bias,S_soft=shape,gt_heading_deg=np.degrees(y))

CASES = [('nominal',{}),('coherent spikes',{'spike':'coherent'}),('localized spike episode',{'spike':'cluster'}),('reduced samples and tilt',{'n':180,'tilt':.5}),('steep field and strong iron',{'h':12,'soft':.40}),('combined stress',{'n':180,'tilt':.5,'h':12,'soft':.40,'spike':'cluster'})]
if __name__ == '__main__':
    for name, kwargs in CASES:
        pairs = [synthetic(i,**kwargs) for i in range(20)]
        report(name,[errors(m.calibrate_and_estimate_heading(x),t) for x,t in pairs])

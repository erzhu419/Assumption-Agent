import pickle
import numpy as np
from scipy.optimize import nnls
import prony_rheology as p

with open('public_rheology.pkl', 'rb') as f:
    public = pickle.load(f)
e = public['experiment']
t,s,y,u,v = (e[k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain'))
tau = np.geomspace(.02, 20, 60)
keys = ('grid',) + tuple((power, strength) for power in (1, 2) for strength in (3, 6, 12, 24))
results = {k: [] for k in keys}

def compare(label,t,s,y,targets,gtrue):
    a = p._design(t,s,tau)
    d = np.c_[np.ones_like(t),(t-t.mean())/np.ptp(t)]
    g,r = p._calibrate(a,d,y)
    keep = np.abs(r-np.median(r)) <= 4.5*p._noise(r)
    q,_ = np.linalg.qr(d[keep],mode='reduced')
    b = y[keep]-q@(q.T@y[keep])
    ap = a[keep]-q@(q.T@a[keep])
    sig = p._noise(r)
    modulus_scale = max(np.linalg.norm(ap@g)/np.linalg.norm(s[keep]-np.mean(s[keep])), np.finfo(float).eps)
    bt = [(p._design(tt,ss,tau), yy) for tt,ss,yy in targets]
    values = []
    for k in keys:
        if k == 'grid':
            gg = g
        else:
            power, strength = k
            row = np.r_[0, (tau/np.ptp(t))**power] * (strength*sig/modulus_scale)
            aa = np.vstack((ap,row))
            norms = np.linalg.norm(aa, axis=0)
            gg = nnls(aa/norms,np.r_[b,0.],maxiter=1200)[0]/norms
        stress_err = max(np.linalg.norm(a1@gg-y1)/np.linalg.norm(y1) for a1,y1 in bt)
        gerr = abs(gg[0]-gtrue)/gtrue
        results[k].append((label,stress_err,gerr,gg[0]))
        values.append(gg[0])
    if label == 'public':
        print('public G by method', dict(zip(map(str,keys),np.round(values,4))))
        print('public stress % by method', [round(100*results[k][-1][1],4) for k in keys])

compare('public',t,s,y,[(u,v,public['gt_test_stress'])], public['gt_G_inf'])
step = .035*(np.clip((u-1)/.15,0,1)-1.4*np.clip((u-9)/.1,0,1)+.6*np.clip((u-17)/.1,0,1))
rng = np.random.default_rng(719)
for slow in (3.,6.,9.,13.,18.):
    for j in range(16):
        gtrue = rng.uniform(8,24)
        times = np.sort(np.r_[rng.uniform(.07,.5), rng.uniform(.75,2.3), slow])
        amp = rng.uniform([16,8,4],[36,24,16])
        if j % 3 == 0:
            times,amp = np.r_[rng.uniform(.022,.055),times],np.r_[rng.uniform(5,18),amp]
        gg = np.r_[gtrue,amp]
        dt = [.04,.05,.075,.10][j % 4]
        tt = np.arange(0,30+dt/2,dt)
        strain = np.where(tt<=14, .05*np.minimum(tt/rng.uniform(1.4,3.4),1), .05+.03*np.sin(rng.uniform(.2,.55)*(tt-14)))
        yy = p._design(tt,strain,times)@gg + rng.uniform(-3,3)+rng.uniform(-.2,.2)*tt + rng.normal(0,[.008,.016,.025,.035][j%4],len(tt))
        ii = rng.choice(len(tt),int(.02*len(tt)),replace=False)
        yy[ii] += rng.uniform(.4,5,len(ii))*rng.choice([-1,1],len(ii))
        targets = [(u,w,p._design(u,w,times)@gg) for w in (v,step)]
        compare(slow,tt,strain,yy,targets,gtrue)

print('Synthetic results; each bin: median modulus error %, 90th percentile %, worst %, modulus failures; final columns: worst stress %, stress failures')
for k, rows in results.items():
    def cell(slow):
        z = np.array([r[2] for r in rows if r[0] == slow]); return (*np.round(100*np.quantile(z,[.5,.9,1]),1),int(sum(z>.06)))
    stress = np.array([r[1] for r in rows[1:]])
    print(k, [cell(slow) for slow in (3,6,9,13,18)], round(100*max(stress),2),int(sum(stress>.022)))

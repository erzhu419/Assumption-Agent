import pickle,numpy as np
from scipy.optimize import root
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; a=np.load('/tmp/feat15.npz'); old=a['z']; p=s['panel_a']; u,v=p['u'],p['v']; L=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]; D=lambda *a:np.c_[np.ones(len(a[0])),*a]; f=lambda z:.18*np.tanh(1.8*z)-.32*np.sin(2.4*np.roll(z,2,axis=-1)); yy=v-f(u); good=np.max(abs(v-u),axis=1)<3
for _ in range(5):
 c=np.asarray([L(D(u[good,j]),yy[good,j]) for j in range(4)]); e=np.linalg.norm(u*c[:,1]+c[:,0]-yy,axis=1);good=e<max(.001,5*np.median(e))
print('A good',sum(good),'coef',c,'inlier rms',np.std(e[good])); pred=lambda z:z*c[:,1]+c[:,0]+f(z); z=np.array([root(lambda q:pred(q)-r,q0,tol=1e-12).x for r,q0 in zip(tr['x'],old)]); print('old inversion residual',np.sort(np.max(abs(pred(old)-tr['x']),axis=1))[-15:],'new residual',np.max(abs(pred(z)-tr['x']))); print('changed rows',np.flatnonzero(np.max(abs(z-old),axis=1)>.001))
p=s['panel_b']; q=s['panel_c']; b=lambda w:D(np.cos(2.6*w)); cc=lambda w:D(w*np.roll(w,-1,axis=1)); w=b(z)@L(b(p['u']),p['v']); h=cc(w)@L(cc(q['u']),q['v1']); dot=np.sum(z*h,axis=1); yz=np.c_[tr['y'],tr['traj']]; n=52
for d in [D(z,h,dot),D(z,h,dot,z[:,1]*h[:,0]),D(z,h,dot,z[:,1]*h[:,0],w)]:
 p=d@L(d[:n],yz[:n]); print('mae',np.mean(abs(p[:n]-yz[:n]),axis=0),np.mean(abs(p[n:]-yz[n:]),axis=0));print('y worst',sorted(enumerate(abs(p[:,0]-yz[:,0])),key=lambda r:-r[1])[:12]);print('coef',L(d[:n],yz[:n])[:,0])
np.savez('/tmp/clean15.npz',z=z,w=w,h=h)
print('eval times',s['eval_times'])

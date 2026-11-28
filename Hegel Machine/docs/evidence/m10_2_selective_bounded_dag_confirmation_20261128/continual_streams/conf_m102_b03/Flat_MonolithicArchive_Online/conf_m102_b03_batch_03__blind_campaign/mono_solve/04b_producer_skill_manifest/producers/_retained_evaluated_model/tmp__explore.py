import importlib.util,numpy as np,pickle
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('arc','/tmp/hegel_continual_archive/conf_m102_b03_batch_00_solver.py'); a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; N=52; one=a._ones; ls=a._ls
np.set_printoptions(precision=5,suppress=True)
u=s['panel_a']['u'];v=s['panel_a']['v']; c=a._align(s['panel_a']); print('time/telemetry',s['eval_times'],s['telemetry_vec']); print('calibration',c,'diagonal error quantiles',np.quantile(abs(v-c[0]-c[1]*u),[.5,.8,.9,.95,.99,1],axis=0)); print('full affine',ls(one(u),v)); print('inverse diag cv archive', [np.mean(abs(p-t[k][N:])) for p,k in zip(a._predict(s,slice(N),t['x'][N:]),['y','traj'])]);
funcs={'cos':np.cos,'sin':np.sin,'tanh':np.tanh,'softsign':lambda w:w/(1+abs(w)),'silu':lambda w:w/(1+np.exp(-w)),'ral':lambda w:w/(1+w*w),'rat':lambda w:1/(1+w*w),'atan':np.arctan,'sq':lambda w:w*w,'cube':lambda w:w**3,'abs':np.abs,'exp':lambda w:np.exp(-w*w)}
p=s['panel_c'];u=p['u'];z=(t['x']-c[0])/c[1]; fs={}
for k in ['v1','v2']:
 v=p[k]; candidates=[]
 for name,f in funcs.items():
  for linear in (False,True):
   def design(w,q):
    fz=f(w*q[0]+q[1:]); return one(np.c_[w,fz] if linear else fz)
   def res(q):
    F=design(u,q); return (F@ls(F,v)-v).ravel()
   qq=min((least_squares(res,np.r_[h,np.zeros(4)],bounds=([.1]+[-4]*4,[4]+[4]*4),max_nfev=100) for h in [.6,1.3,2.3]),key=lambda fit:fit.cost);pred=design(z,qq.x)@ls(design(u,qq.x),v); candidates.append((np.sqrt(np.mean(qq.fun**2)),name,linear,qq.x,pred))
 candidates.sort(key=lambda v:v[0]);fs[k]=candidates[0][-1];print(k,[(round(i[0],7),i[1],i[2],i[3]) for i in candidates[:9]])
B=s['panel_b']; b=one(z)@ls(one(B['u']),B['v']); c,d=fs.values();
for label,F in {'raw':t['x'],'z':z,'b':b,'c':c,'d':d,'cd':np.c_[c,d],'bcd':np.c_[b,c,d],'ALL':np.c_[z,c,d], 'cdot':np.c_[c,d,np.sum(c*d,axis=1)],'cxd':np.c_[c,d,(c[:,:,None]*d[:,None,:]).reshape(len(z),-1)],'ccdd':np.c_[c,d,(c[:,:,None]*c[:,None,:]).reshape(len(z),-1),(d[:,:,None]*d[:,None,:]).reshape(len(z),-1)]}.items():
 F=one(F);print(label, [tuple(round(w,5) for w in (np.mean(abs(F[N:]@ls(F[:N],t[k][:N])-t[k][N:])),np.mean(abs(F[:N]@ls(F[:N],t[k][:N])-t[k][:N])))) for k in ['y','traj']])
np.savez('/tmp/ops.npz',x=t['x'],z=z,b=b,c=c,d=d,y=t['y'],traj=t['traj'])

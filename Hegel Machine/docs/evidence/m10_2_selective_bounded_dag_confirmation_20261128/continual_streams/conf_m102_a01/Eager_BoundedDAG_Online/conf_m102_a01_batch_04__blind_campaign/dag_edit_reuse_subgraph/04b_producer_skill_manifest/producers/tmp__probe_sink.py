import sys, pickle, numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00,node_01,node_02
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=st['train']; x,y,t=(tr[k] for k in ('x','y','traj')); n=len(y)
print('study keys',list(st),'telemetry',st.get('telemetry_vec'),'summary',tr.get('summary'))
for seed in range(17):
 if seed==0: a,b=np.arange(n-20),np.arange(n-20,n)
 else:
  p=np.random.default_rng(seed).permutation(n); a,b=p[:-20],p[-20:]
 s=dict(st,train=dict(x=x[a],y=y[a],traj=t[a],summary=np.array([y[a].mean(),y[a].std(),t[a,-1].mean(),t[a,-1].std()])),test=dict(x=x[b]))
 shared=node_00.run_node({'__input__':s}); pp=node_01.run_node({'node_00':shared})['poly']; sk=node_02.run_node({'node_00':shared}); kk=sk['smooth']; vt=sk['vt']; te=(.4*pp[:,1:]+.6*kk[:,1:])@vt
 if seed==0: print('public rows y/poly/smooth',np.round(np.c_[y[b],pp[:,0],kk[:,0]],4))
 np.savez('/tmp/sink_fold_%02d.npz'%seed,p=pp,k=kk,y=y[b],t=t[b],te=te,vt=vt,a=a,b=b)
 d=pp[:,0]-kk[:,0]; base=kk[:,0]; mae=lambda u: round(np.abs(u-y[b]).mean(),5)
 print(seed,[mae(base+w*d) for w in [-.5,0,.25,.5,.65,.75,1,1.25,1.5]],'one-sided',[mae(base+c*np.maximum(d,0)+e*np.minimum(d,0)) for c,e in [(1,0),(0,1),(.65,0),(0,.65)]],'traj',round(np.abs(te-t[b]).mean(),5),flush=True)

exec(open('/tmp/probe.py').read().split('m=old._panel_maps')[0])
m=load('/holdout_task/node_00.py'); Y=s['train']['y']; T=s['train']['traj'];
for held in (np.arange(52,72),np.arange(0,20),np.arange(20,40),np.arange(40,60)):
 tr=np.setdiff1d(np.arange(len(x)),held); sub=dict(s);sub['train']=dict(x=x[tr],y=Y[tr],traj=T[tr]);sub['test']={'x':x[held]}
 c,e=m._coordinates(sub,x[tr],x[held]);o=sink.run_node({'__input__':sub,'node_00':{'c':c[:len(tr)],'query':c[len(tr):]}});print(held[0],e,'raw scalar',np.mean(abs(c[len(tr):,0]-Y[held])),'DAG',np.mean(abs(o['y_hat']-Y[held])),np.mean(abs(o['state_pred']-T[held])))

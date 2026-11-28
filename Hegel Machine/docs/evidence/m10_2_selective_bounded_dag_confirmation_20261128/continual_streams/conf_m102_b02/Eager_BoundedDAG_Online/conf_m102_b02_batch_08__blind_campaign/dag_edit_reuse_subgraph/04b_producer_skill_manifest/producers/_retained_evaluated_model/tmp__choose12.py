import sys,pickle,numpy as np
sys.path.insert(0,'/tmp')
import scalar12 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['y']); ix=np.arange(n)
x=m.features(s,tr['x']); ys=tr['y']; errs=[]; stress=[]
for it in (ix[52:],ix[:20],ix[20:40],ix[40:60],ix[::4],ix[1::4],ix[2::4],ix[3::4]):
    ir=np.setdiff1d(ix,it); f=[a[np.r_[ir,it]] for a in x]; y=ys[ir]; truth=ys[it]
    pred=m.scalar_operators(*f,y)
    a=list(m.operators(*f,y)); legacy=m.select(*f,y,'sparse',None)@y
    ref=[next(r[2]@y for r in a if r[0]==key) for key in [(('sym+',3,3.),.005),(('d+',3,3.),.005),(('bd+',3,3.),.04)]]
    pred=[legacy]+pred+[np.mean(ref,axis=0)]+[(legacy+r)/2 for r in pred[1:]]+[(np.mean(ref,axis=0)+r)/2 for r in pred[1:]]
    er=np.mean(abs(np.array(pred)-truth),axis=1); errs.append(er); print('scalar slice',it[0],np.round(er,5),flush=True)
e=np.array(errs); worst=e[1:].max(0); score=e[1:].mean(0)+.35*worst
valid=np.flatnonzero((e[0]<=.038) | ((e[0]<e[0,0]) & (worst<=worst[0]+.005)))
j=int(min(valid,key=lambda i:(e[0,i]>.038,score[i])) if len(valid) else 0)
print('selected scalar',j,'official',e[0,j],'worst alternate',worst[j],flush=True)
if j:
    p='/holdout_task/node_00.py'; source=open('/tmp/scalar12.py').read(); source=source.replace("    t = select(z,b,c,d,h,tr['traj'],FAMILY_T,FIXED_T)",f"    y = final_scalar(z,b,c,d,h,tr['y'],y,{j})\n    t = select(z,b,c,d,h,tr['traj'],FAMILY_T,FIXED_T)")
    source+='''


def final_scalar(z,b,c,d,h,y,legacy,j):
    pred=scalar_operators(z,b,c,d,h,y)
    if 1 <= j <= 5:
        return pred[j-1]
    if 7 <= j <= 10:
        return (legacy+pred[j-6])/2
    keys=[(('sym+',3,3.),.005),(('d+',3,3.),.005),(('bd+',3,3.),.04)]
    ref=np.mean([next(operators(z,b,c,d,h,y,fixed=k))[2]@y for k in keys],axis=0)
    return ref if j==6 else (ref+pred[j-10])/2
'''
    open(p,'w').write(source)

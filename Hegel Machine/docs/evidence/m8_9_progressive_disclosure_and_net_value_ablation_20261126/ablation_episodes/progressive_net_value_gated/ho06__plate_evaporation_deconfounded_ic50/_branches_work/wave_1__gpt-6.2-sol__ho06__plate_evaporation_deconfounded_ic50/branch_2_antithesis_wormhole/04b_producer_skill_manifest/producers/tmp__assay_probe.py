import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

with open('/holdout_task/public_screen.pkl','rb') as f:
    data=pickle.load(f)
plates=data['plates']

def geom(ws):
    r=np.array([w['row'] for w in ws]); c=np.array([w['col'] for w in ws]);
    return np.array([r,15-r,c,23-c]).T

def design(d,t,kind):
    r=d[:,0]/7.5-1; c=d[:,2]/11.5-1
    e=np.exp(-d/t)
    nearest=np.max(e,axis=1)
    s=e.sum(axis=1)
    cols=[np.ones(len(d)),r,c]
    if kind=='constant': return np.ones((len(d),1))
    if kind=='nearest': cols += [nearest]
    if kind=='sum': cols += [s]
    if kind=='union': cols += [s,(e[:,0]+e[:,1])*(e[:,2]+e[:,3])]
    if kind=='axes': cols += [e[:,0]+e[:,1],e[:,2]+e[:,3]]
    if kind=='mixed': cols += [nearest,s-nearest]
    return np.column_stack(cols)

D=[]; Y=[]; Q=[]; C=[]
for p in plates:
    w=p['wells']; dm=[x for x in w if x['well_type']=='DMSO']; pk=[x for x in w if x['well_type']=='POS_KILL']; cm=[x for x in w if x['well_type']=='COMPOUND']
    b=np.median([x['fluorescence'] for x in pk]); yd=np.array([x['fluorescence']-b for x in dm]); s=np.median(yd); D.append(geom(dm)); Y.append(yd/s); Q.append(geom(cm)); C.append((cm,np.array([(x['fluorescence']-b)/s for x in cm])))
print('DMSO well positions and across-plate median relative signal')
for ij,y in zip(D[0][:,[0,2]],np.median(Y,axis=0)): print(tuple(ij),round(y,4))
print('Compound distance support:',np.unique(np.concatenate(Q).min(axis=1),return_counts=True))

def reg(X,y):
    a=np.linalg.lstsq(X,y,rcond=None)[0]
    return least_squares(lambda k:X@k-y,a,loss='soft_l1',f_scale=.018).x

def score(t,kind):
    oo=[]; cc={}
    for d,y,q,(cm,cy) in zip(D,Y,Q,C):
        x=design(d,t,kind); fitted=reg(x,y); v=cy/(design(q,t,kind)@fitted)
        for i in range(len(y)):
            take=np.arange(len(y))!=i
            oo.append(y[i]-x[i]@reg(x[take],y[take]))
        for w,z in zip(cm,v): cc.setdefault(w['compound_id'],[]).append((w['log10_conc_uM'],z))
    err=[]
    for name,rows in sorted(cc.items()):
        a,v=np.array(rows).T
        fit=least_squares(lambda k:expit(np.log(10)*k[1]*(k[0]-a))-v,[np.median(a),1.],bounds=([-10,.05],[10,8]),loss='cauchy',f_scale=.026).x
        gt=data['gt'][name]; err.append(fit-[gt['log10_ic50_uM'],gt['hill_slope']])
    e=np.abs(err); oo=np.abs(oo)
    return np.sqrt(np.mean(oo**2)),np.max(oo),*np.max(e,axis=0),np.argmax(e,axis=0).tolist()

for k in ['constant','nearest','sum','union','axes','mixed']:
    for t in ([1.] if k=='constant' else [.65,1.,1.4,1.8,2.3,3.2]):
        rms,mx,ic,h,who=score(t,k)
        print(f'{k:8s} t={t:4.2f} DMSO LOO rms/max={rms:.4f}/{mx:.4f} max IC50/slope errors={ic:.4f}/{h:.4f} compounds={who}')

import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

with open('/holdout_task/public_screen.pkl','rb') as f:
    public = pickle.load(f)
plates = []
for plate in public['plates']:
    wells = plate['wells']
    xy = np.array([(w['row'],w['col']) for w in wells])
    y = np.array([w['fluorescence'] for w in wells])
    kind = np.array([w['well_type'] for w in wells])
    b = np.median(y[kind == 'POS_KILL'])
    plates.append((wells, xy, y-b, kind == 'DMSO', kind == 'COMPOUND'))

print('Paired DMSO locations and background-subtracted signal ratios')
ref = np.median(plates[0][2][plates[0][3]])
for xy,s in zip(plates[0][1][plates[0][3]], plates[0][2][plates[0][3]]):
    r,c=xy
    vals=[]
    for _,p,z,d,_ in plates:
        iz=d & (p==xy).all(axis=1)
        vals.append(round(float(z[iz][0]/np.median(z[d])),3) if iz.any() else None)
    print((r,c),vals)

def features(xy, model, decay):
    r,c = xy.T
    u,v = (r-7.5)/7.5,(c-11.5)/11.5
    dr,dc = np.minimum(r,15-r),np.minimum(c,23-c)
    e = np.exp(-np.minimum(dr,dc)/decay)
    er,ec = np.exp(-dr/decay),np.exp(-dc/decay)
    es = np.exp(-r/decay)+np.exp(-(15-r)/decay)+np.exp(-c/decay)+np.exp(-(23-c)/decay)
    cols = {'scalar':[], 'linear':[u,v], 'min':[e,u,v], 'minQ':[e,u,v,u*u,v*v,u*v],
            'sum':[es,u,v], 'sumQ':[es,u,v,u*u,v*v,u*v], 'axis':[er,ec,u,v],
            'axisQ':[er,ec,u,v,u*u,v*v,u*v], 'union':[er,ec,er*ec,u,v],
            'border':[np.minimum(dr,dc)==0,u,v], 'quadratic':[u,v,u*u,v*v,u*v]}[model]
    return np.column_stack([np.ones(len(r)),*cols])

def reg(A,y):
    x=np.linalg.lstsq(A,y,rcond=None)[0]
    return least_squares(lambda v:A@v-y,x,loss='soft_l1',f_scale=.012).x

def hills(records):
    errors=[]
    for name in sorted(records):
        c,v=np.array(records[name]).T
        fit=least_squares(lambda v0: expit(-np.log(10)*np.exp(v0[1])*(c-v0[0]))-v,[np.median(c),0.],loss='soft_l1',f_scale=.025,bounds=([-10,-3],[10,3]))
        truth=public['gt'][name]
        errors.append([abs(fit.x[0]-truth['log10_ic50_uM']),abs(np.exp(fit.x[1])-truth['hill_slope'])])
    e=np.array(errors)
    return np.round(e.max(axis=0),4),np.round(e.mean(axis=0),4)

for model in ('scalar','linear','quadratic','border','min','minQ','sum','sumQ','axis','axisQ','union'):
    for decay in ((.65,1.,1.5,2.,3.,5.) if model in ('min','minQ','sum','sumQ','axis','axisQ','union') else (1.,)):
        for pool in (False,True):
            data=[]
            for i,(w,xy,y,d,t) in enumerate(plates):
                scale=np.median(y[d]); A=features(xy,model,decay)
                if pool:
                    A=np.column_stack([np.broadcast_to(np.eye(len(plates))[i],(len(y),len(plates))),A[:,1:]])
                data.append((w,xy,y/scale,d,t,A))
            records={}; miss=[]
            groups=[data] if pool else [[p] for p in data]
            for group in groups:
                X=np.concatenate([A[d] for _,_,_,d,_,A in group]); Z=np.concatenate([y[d] for _,_,y,d,_,_ in group]); P=np.concatenate([xy[d] for _,xy,_,d,_,_ in group]);
                for pos in np.unique(P,axis=0):
                    test=(P==pos).all(axis=1); train=~test
                    miss.extend(zip(P[test],(Z[test]-X[test]@reg(X[train],Z[train]))/Z[test]))
                beta=reg(X,Z)
                for w,_,y,_,t,A in group:
                    for j in np.flatnonzero(t):
                        x=w[j]; records.setdefault(x['compound_id'],[]).append((x['log10_conc_uM'],y[j]/max(.1,A[j]@beta)))
            edge=np.array([min(r,15-r,c,23-c)==0 for (r,c),e in miss]); er=np.array([e for _,e in miss]); scores=[np.sqrt((er[s]**2).mean()) for s in (edge,~edge)]
            print(model,decay,int(pool),'ctrl E/I RMS%',*(round(100*v,2) for v in scores),'Hill max/mean',*hills(records))

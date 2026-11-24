import os
os.makedirs('/holdout_task',exist_ok=True)
code=r'''import numpy as np
from scipy.optimize import least_squares

def _rot_level(v, rp, signs=(1,1)):
    # Inverse roll/pitch rotation: body vector -> NED-level axes.
    phi, theta = rp[:,0], rp[:,1]
    p, t = signs[0]*phi, signs[1]*theta
    cp,sp,ct,st=np.cos(p),np.sin(p),np.cos(t),np.sin(t)
    # R_y(theta) R_x(phi) applied to body-to-level (transpose of body_from_ned)
    x=ct*v[:,0]+st*sp*v[:,1]+st*cp*v[:,2]
    y=cp*v[:,1]-sp*v[:,2]
    return np.column_stack((x,y))

def _fit(raw):
    # Corrected vector = exp(H) (raw-b); symmetric H guarantees SPD.
    raw=np.asarray(raw,float)
    med=np.median(raw,axis=0)
    scale=np.maximum(np.std(raw,axis=0),1.)
    def mat(q):
        h=np.array([[q[3],q[6],q[7]],[q[6],q[4],q[8]],[q[7],q[8],q[5]]])
        w,u=np.linalg.eigh(h); return (u*np.exp(w))@u.T
    def pred(q):
        A=mat(q); return np.linalg.norm((raw-q[:3])@A.T,axis=1)
    q0=np.r_[med,0.,0.,0.,0.,0.,0.]
    # robust norm-radius fit; nuisance radius is median corrected norm.
    def residual(q):
        r=pred(q); radius=np.median(r); return r-radius
    opt=least_squares(residual,q0,loss='soft_l1',f_scale=.7,max_nfev=300)
    A=mat(opt.x)
    # det(S)=1 fixes calibration scale; A=S^-1 up to field magnitude, set det(A)=1.
    A=A/np.linalg.det(A)**(1/3)
    return opt.x[:3],A

def calibrate_and_estimate_heading(session: dict) -> dict:
    raw=np.asarray(session['calib_m_raw'],float)
    b,A=_fit(raw)
    test=np.asarray(session['test_m_raw'],float); rp=np.asarray(session['test_roll_pitch_rad'],float)
    corrected=(test-b)@A.T
    # Standard NED leveling, heading atan2(-east,north).
    level=_rot_level(corrected,rp)
    heading=np.degrees(np.arctan2(-level[:,1],level[:,0]))
    heading=(heading+180.)%360.-180.
    heading=np.where(heading<=-180.,180.,heading)
    return {'S_soft_hat':np.linalg.inv(A),'b_hard_hat':b,'heading_deg':heading}
'''
open('/holdout_task/mag_calibrator.py','w').write(code)

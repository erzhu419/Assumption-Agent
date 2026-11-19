#!/usr/bin/env python3
"""Analytic illustrations for a proposed transfer hypothesis; not Hegel runs."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    # A third-domain, analyst-specified linear mechanical model.
    k1, k2, eta = 3.0, 2.0, 10.0
    tau = eta / (k1 + k2)
    plateau = k1 * k2 / (k1 + k2)
    initial = k1
    delta = 1.0
    samples = [plateau + (initial-plateau)*math.exp(-i*delta/tau) for i in range(4)]
    q = (samples[2]-samples[1])/(samples[1]-samples[0])
    fitted_tau = -delta/math.log(q)
    fitted_plateau = (samples[1]-q*samples[0])/(1-q)
    predicted_fourth = samples[2]+q*(samples[2]-samples[1])
    assert abs(fitted_tau-tau) < 1e-12
    assert abs(fitted_plateau-plateau) < 1e-12
    assert abs(predicted_fourth-samples[3]) < 1e-12

    time_grid = [i/100 for i in range(801)]
    step_rows = [{'theta':t,'chemical_conjugate_normalized':0.4+0.6*math.exp(-t),
                  'rl_conjugate_normalized':math.exp(-t),
                  'rl_total_flux_normalized':1-math.exp(-t)} for t in time_grid]
    pulse_rows=[]
    for t in time_grid:
        ext=t*t*math.exp(-t)
        net=t*t*t*math.exp(-t)/3
        ind=net-ext
        ext_dot=math.exp(-t)*(2*t-t*t)
        net_dot=math.exp(-t)*(t*t-t*t*t/3)
        current=ind  # L=1 in normalized units
        current_dot=net_dot-ext_dot
        residual=current_dot+current+ext_dot
        energy_residual=current*current_dot-(-ext_dot*current-current*current)
        pulse_rows.append({'theta':t,'external_flux':ext,'total_flux':net,'self_flux':ind,
                           'net_over_external':net/ext if ext else None,
                           'rl_residual':residual,'energy_residual':energy_residual})
    assert max(abs(x['rl_residual']) for x in pulse_rows) < 1e-12
    assert max(abs(x['energy_residual']) for x in pulse_rows) < 1e-12
    assert pulse_rows[400]['net_over_external'] > 1
    result={
      'scope':'Analyst-specified analytic models and algebraic checks. No autonomous hypothesis induction, external laboratory, Hegel performance study, or baseline timing.',
      'step_plot':{'theta':'t/tau for each normalized model','chemical_rho':0.6,'rl_rho':1.0,
                   'protocol':'zero increments before a step, then input held; signed conjugate output compared with frozen internal-state response',
                   'rows':step_rows},
      'pulse_plot':{'external':'theta**2 * exp(-theta)','net':'theta**3 * exp(-theta)/3',
                    'L':1.0,'R':1.0,'A':1.0,'zero_initial_current':True,'rows':pulse_rows},
      'mechanical_example':{'k1':k1,'k2':k2,'eta':eta,'unit_step':1.0,'delta_t':delta,
                'true_tau':tau,'true_rho':k1/(k1+k2),'true_plateau':plateau,
                'samples':[{'t':i*delta,'p':y,'use':'fit' if i<3 else 'unfitted_prediction_check'} for i,y in enumerate(samples)],
                'q_fitted':q,'tau_fitted':fitted_tau,'plateau_fitted':fitted_plateau,
                'fourth_predicted':predicted_fourth,'fourth_error':abs(predicted_fourth-samples[3]),
                'caution':'The fourth point is held out from fitting within the same trajectory, not an independent episode. Noiseless nonconstant single-mode data are assumed. Input/output fit does not identify a unique hidden physical realization.'}}
    (ROOT/'illustrative_models.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.8))
    a,b=axes
    t=time_grid
    a.plot(t,[r['chemical_conjugate_normalized'] for r in step_rows],label='Chemical conjugate response (rho = 0.6)',color='#2563eb',lw=2.4)
    a.plot(t,[r['rl_conjugate_normalized'] for r in step_rows],label='RL conjugate response: -I (rho = 1)',color='#0d9488',lw=2.4)
    a.plot(t,[r['rl_total_flux_normalized'] for r in step_rows],label='RL total flux: a different output',color='#d97706',lw=2.2,ls='--')
    a.axhline(1,color='#94a3b8',ls=':',lw=1)
    a.set(title='A. Align the output before comparing',xlabel='Normalized time t / tau',ylabel='Normalized step response',xlim=(0,7),ylim=(-0.03,1.08))
    a.legend(loc='center right',fontsize=8.5,frameon=True,framealpha=.94)
    ext=[r['external_flux'] for r in pulse_rows]
    net=[r['total_flux'] for r in pulse_rows]
    b.plot(t,ext,label='External flux',color='#475569',lw=2.4)
    b.plot(t,net,label='Total flux in a passive RL circuit',color='#7c3aed',lw=2.4)
    b.fill_between(t,ext,net,where=[v>=3 for v in t],color='#c4b5fd',alpha=.35,label='Total flux exceeds current input')
    idx=400
    b.scatter([4],[net[idx]],color='#7c3aed',s=35,zorder=4)
    b.annotate('At t = 4 tau:\ntotal / external = 4/3',xy=(4,net[idx]),xytext=(4.7,.30),
               fontsize=9,arrowprops={'arrowstyle':'->','color':'#7c3aed'},color='#5b21b6')
    b.set(title='B. Memory defeats a pointwise attenuation bound',xlabel='Normalized time t / tau',ylabel='Flux / pulse scale',xlim=(0,8),ylim=(-.015,.66))
    b.legend(loc='upper right',fontsize=8.5,frameon=True,framealpha=.94)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle('Analytic illustrations of the proposed hypothesis and its scope',fontsize=14,fontweight='bold',y=.98)
    fig.text(.5,.015,'Calculated from specified models; not a Hegel benchmark or an autonomous discovery result.',ha='center',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.04,1,.94))
    fig.savefig(ROOT/'response_comparison.png',dpi=190,bbox_inches='tight')
    fig.savefig(ROOT/'response_comparison.svg',bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'mechanical_fit':result['mechanical_example'],
                      'pulse_at_t4':pulse_rows[400],
                      'figure':'response_comparison.png'},ensure_ascii=False))


if __name__=='__main__':
    main()

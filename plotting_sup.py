
import matplotlib.pyplot as plt
import numpy as np
import figure_properties as fp
from utils import Q_nak, Recorder
from mitochondria import Mito

dt = 0.01

def single_spike(spike_at=150, q=10, t_start=150, t_run=5000, t_ca1=7000, t_ca2=20000):
    tt = np.arange(0, t_run, 0.01)
    spike_costs = np.zeros_like(tt)
    ca_matrix = np.zeros_like(tt)
    ca_matrix2 = np.zeros_like(tt)
    Q_val = Q_nak(tt, q, tau_Q=100)  # ATP expenditure increase per-spike
    ca_influx = Q_nak(tt, fact=0.1, tau_Q=t_ca1)
    ca_influx2 = Q_nak(tt, fact=0.1, tau_Q=t_ca2)
    t_idx = np.where(tt > spike_at)[0][1] - 1
    spike_costs[t_idx:] += Q_val[:len(spike_costs[t_idx:])]
    ca_matrix[t_idx:] += ca_influx[:len(ca_matrix[t_idx:])]
    ca_matrix2[t_idx:] += ca_influx2[:len(ca_matrix2[t_idx:])]
    return t_run, tt, spike_costs, ca_matrix, ca_matrix2


def spike_train_costs(time, tt, baseline_atp, spike_costs, ca_matrix):
    '''Perturbations due to a spike train'''
    factor = 0.1e-4  # PSI leak factor for Ca per-spike
    m = Mito(baseline_atp=baseline_atp)
    m.steadystate_vals(time=2000)  # steady state - wait till we reach here
    rec_vars_list = ['atp', 'ca_mat', 'k_ant']
    m_record = Recorder(m, rec_vars_list, time, dt)
    for ii, tt in enumerate(np.arange(0, time, dt)):
        try:
            m.update_vals(dt,
                          atp_cost=spike_costs[ii],  # per-spike cost
                          leak_cost=spike_costs[ii]*factor,  # ca influx cost
                          ca_mat=ca_matrix[ii])  # excess free ca in matrix
        except IndexError:
            m.update_vals(dt, leak_cost=0,
                          atp_cost=0, ca_mat=0)
        m_record.update(ii)
    return m_record

baseline_atp = 100
time, tt, spike_vals, ca_cntrl, ca_letm1 = single_spike()
rec_cntrl = spike_train_costs(time, tt, baseline_atp, spike_vals, ca_cntrl)


layout = [["A", "C"],
          ["B", "D"]]
fig, axd = plt.subplot_mosaic(layout,
                              figsize=fp.cm_to_inches([10, 9.14]),
                              width_ratios=[1, 1],
                              height_ratios=[1, 1])

#axd['A'].plot(tt, rec_cntrl.out['k_ant']*1000)
axd['A'].fill_between(tt, rec_cntrl.out['k_ant']*1000, color='#4daf4a')
axd['A'].fill_between(tt, [baseline_atp]*len(tt), color='wheat')
axd['A'].plot([150, 150], [0, 125], ls='--', c='k', lw=1)
axd['A'].set_ylim(0, 125)
axd['A'].set_xlim(100, 300)
axd['A'].set_ylabel(r'$K_{ANT}$' + ' /ks')
axd['A'].set_xlabel('Time (ms)')
axd['A'].text(200, 50, s='Non-spiking \ncosts', ha='center', va='center', fontsize=7)
axd['A'].text(200, 120, s='Per-spike cost', ha='center', va='center', fontsize=7)

#axd['B'].plot(tt, rec_cntrl.out['atp'], c='k')
axd['B'].set_ylim(0, 1)
axd['B'].set_xlim(100, 300)
axd['B'].set_ylabel('ATP (a.u)')
axd['B'].set_yticks([0., 1])
axd['B'].set_yticklabels([0., 1])
axd['B'].fill_between(tt, rec_cntrl.out['atp'], color='gold')
axd['B'].fill_between(tt, rec_cntrl.out['atp'], 1, color='silver')
axd['B'].plot([150, 150], [0, 1], ls='--', c='k', lw=1)
axd['B'].set_xlabel('Time (ms)')
axd['B'].text(200, 0.3, s='ATP', ha='center', va='center', fontsize=7)
axd['B'].text(200, 0.75, s='ADP', ha='center', va='center', fontsize=7)


axd['C'].plot(tt, rec_cntrl.out['ca_mat'], 'k', label='Control', zorder=10)
axd['C'].plot(tt, ca_letm1, c='#FF0A6C', label='LETM1KD', zorder=9)
axd['C'].plot(tt, ca_letm1*0, c='#00bfff', label='MCUKD', zorder=8)
axd['C'].legend(frameon=False, ncols=1, loc='upper center', title='Single spike')
axd['C'].set_ylim(-0.05, 0.4)
axd['C'].set_ylabel('$Ca_{mito}$ (a.u)')
axd['C'].set_yticks([0, 0.2, 0.4])
axd['C'].set_xlabel('Time (ms)')


ca_mat = np.arange(0, 20, 0.005)
fracs = 10 / (1 + np.exp(-0.5*(ca_mat-9)))
axd['D'].plot(ca_mat, fracs, c='k')
axd['D'].set_xlabel('$Ca_{mito}$ (a.u)')
axd['D'].set_ylabel('frac (a.u)')
axd['D'].plot(1.2, -0.5, marker='*', c='k',
              clip_on=False, markersize=7.5, linestyle='',
              markeredgecolor='none', label='Control')
axd['D'].plot(3.4, -0.5, marker='*', c='#FF0A6C', linestyle='',
              clip_on=False, markersize=7.5, zorder=10,
              markeredgecolor='none', label='LETM1KD')
axd['D'].plot(0, -0.5, marker='*', c='#00bfff',
              clip_on=False, markersize=7.5, linestyle='',
              markeredgecolor='none', label='MCUKD', zorder=10)
axd['D'].legend(frameon=False, ncols=1, loc='lower right', title='2Hz Random')
axd['D'].set_ylim(-0.5, 10.5)

fp.remove_spines([axd['A'], axd['B'], axd['C'], axd['D']],
                 bottom=False, left=False)
# plt.show()
plt.tight_layout()
plt.savefig('single_spike.svg')

import numpy as np
from scipy import stats
from utils import Recorder, Q_nak
from mitochondria import Mito
import matplotlib.pyplot as plt
from figure_properties import *

# Version will all the necessary parameters.

dt = 0.01
seed = 11
np.random.seed(seed)


def generate_spiketrain(rate, duration):  # duration given in ms
    n_spikes = int(duration*rate/1000)
    isi = stats.expon.rvs(scale=1./rate, size=n_spikes)
    spike_train = 1000*np.cumsum(isi) # returning also in ms
    # spike_train = spike_train[np.where(spike_train < duration)]
    print(spike_train[-1])
    return spike_train


def poi_spiking(sp_hz, q, t_start=150, t_run=1000, t_ca_1=5000, t_ca_2=7000):
    t_silent = t_run
    time = t_start + t_run + t_silent  # ms total sim time, t_start is wait time
    tt = np.arange(0, time, dt)
    spike_train = generate_spiketrain(sp_hz, t_run)
    print(len(spike_train))
    idx_spike = [np.where(ii+t_start < tt)[0][0] for ii in spike_train]
    spike_costs = np.zeros_like(tt)
    ca_matrix_1 = np.zeros_like(tt)
    ca_matrix_2 = np.zeros_like(tt)
    Q_val = Q_nak(tt, q, tau_Q=100)  # ATP expenditure increase per-spike
    ca_influx_1 = Q_nak(tt, fact=0.1, tau_Q=t_ca_1)
    ca_influx_2 = Q_nak(tt, fact=0.1, tau_Q=t_ca_2)
    for t_idx in idx_spike:
        spike_costs[t_idx:] += Q_val[:len(spike_costs[t_idx:])]
        ca_matrix_1[t_idx:] += ca_influx_1[:len(ca_matrix_1[t_idx:])]
        ca_matrix_2[t_idx:] += ca_influx_2[:len(ca_matrix_2[t_idx:])]
    return time, tt, spike_costs, ca_matrix_1, ca_matrix_2, spike_train+150


def spike_train_costs(time, tt, baseline_atp, spike_costs, ca_matrix):
    '''Perturbations due to a spike train'''
    factor = 0.1e-4  # PSI leak factor for Ca per-spike
    m = Mito(baseline_atp=baseline_atp)
    m.steadystate_vals(time=2000)  # steady state - wait till we reach here
    rec_vars_list = ['atp', 'ca_mat', 'jjV',
                     'pyr', 'cit', 'akg', 'acc', 'oaa']
    m_record = Recorder(m, rec_vars_list, time, dt)
    for ii, tt in enumerate(np.arange(0, time, dt)):
        try:
            m.update_vals(dt,
                          atp_cost=spike_costs[ii],  # per-spike cost
                          leak_cost=spike_costs[ii]*factor,  # ca influx cost
                          ca_mat=ca_matrix[ii])  # excess free ca in matrix
                          # ca_mat=0)  # no excess free ca in matrix
        except IndexError:
            m.update_vals(dt, leak_cost=0,
                          atp_cost=0, ca_mat=0)
        m_record.update(ii)
    print(m.fetch_actual_conc())
    return m_record


def dump_vals(filename, rec, tt, train):
    np.savez(filename,
             atp=rec.out['atp'],
             jjV=rec.out['jjV'],
             ca_mat=rec.out['ca_mat'],
             pyr=rec.out['pyr'],
             # cit=rec.out['cit'],
             akg=rec.out['akg'],
             # acc=rec.out['acc'],
             # oaa=rec.out['oaa'],
             tt=tt,
             spike_train=train)

if __name__ == '__main__':
    baseline_atp = 100
    firing_freq = 50
    time, tt, spike_vals, ca_cntrl, ca_letm1, sp_train = poi_spiking(sp_hz=firing_freq,
                                                                     q=10,
                                                                     t_start=150, t_run=100000,
                                                                     t_ca_1=7000, t_ca_2=20000)
    rec_cntrl = spike_train_costs(time, tt, baseline_atp, spike_vals, ca_cntrl)
    rec_letm1 = spike_train_costs(time, tt, baseline_atp, spike_vals, ca_letm1)
    rec_mcukd = spike_train_costs(time, tt, baseline_atp, spike_vals, ca_cntrl*0)

    labels = ['Control', 'Letm1', 'MCUKD']
    t_ca = [7, 20, 7]
    for ii, rec in enumerate([rec_cntrl, rec_letm1, rec_mcukd]):
        filename = str(seed) + '_' + str(baseline_atp) + '_poi_'+str(int(firing_freq))+'Hz_' + str(t_ca[ii]) + '_' + labels[ii] + '.npz'
        dump_vals(filename, rec, tt, sp_train)
        

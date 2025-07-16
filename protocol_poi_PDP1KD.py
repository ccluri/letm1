import numpy as np
from scipy import stats
from utils import Recorder_select, Q_nak
from mitochondria_PDP1KD import Mito
import matplotlib.pyplot as plt
from figure_properties import *

# Version will all the necessary parameters.

dt = 0.1
seed = 11
np.random.seed(seed)


def generate_spiketrain(rate, duration):  # duration given in ms
    n_spikes = int(duration*rate/1000)
    isi = stats.expon.rvs(scale=1./rate, size=n_spikes)
    spike_train = 1000*np.cumsum(isi) # returning also in ms
    spike_train = spike_train[np.where(spike_train < duration)]
    print('last spike occured at (ms): ', spike_train[-1])
    return spike_train


def poi_spiking(sp_hz, q, t_start=150, t_run=1000, t_ca_1=5000, t_ca_2=7000):
    t_silent = 0 # t_run
    time = t_start + t_run + t_silent  # ms total sim time, t_start is wait time
    tt = np.arange(0, time, dt)
    spike_train = generate_spiketrain(sp_hz, t_run)
    print('total number of spikes: ', len(spike_train))
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
    t_start = 150
    factor = 0.1e-4  # PSI leak factor for Ca per-spike
    m = Mito(baseline_atp=baseline_atp)
    m.steadystate_vals(time=2000)  # steady state - wait till we reach here
    rec_vars_list = ['atp', 'pyr']
    m_record = Recorder_select(m, rec_vars_list)
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

        ## selectively record last 10 seconds every [0, 50, 90]
        rec_dur = 10000
        start_rec_at = [0, 90, 190, 390, 790, 1590, 3190] # SEC
        start_rec_at = [ii*1000 for ii in start_rec_at] # ms
        start_rec_at = [ii+t_start for ii in start_rec_at] # adding offset
        if tt > start_rec_at[0] and tt < (rec_dur + start_rec_at[0]):
            m_record.update(tt)
        elif tt > start_rec_at[1] and tt < (rec_dur + start_rec_at[1]):
            m_record.update(tt)
        elif tt > start_rec_at[2] and tt < (rec_dur + start_rec_at[2]):
            m_record.update(tt)
        elif tt > start_rec_at[3] and tt < (rec_dur + start_rec_at[3]):
            m_record.update(tt)
        elif tt > start_rec_at[4] and tt < (rec_dur + start_rec_at[4]):
            m_record.update(tt)
        elif tt > start_rec_at[5] and tt < (rec_dur + start_rec_at[5]):
            m_record.update(tt)
        elif tt > start_rec_at[6] and tt < (rec_dur + start_rec_at[6]):
            m_record.update(tt)
    m_record.convert_to_arrays()
    print(m.fetch_actual_conc())
    return m_record


def dump_vals(filename, rec, tt, train, spike_costs, ca_cntrl, ca_letm1):
    np.savez(filename,
             times=rec.rec_times,
             atp=rec.out['atp'],
             pyr=rec.out['pyr'],
             tt=tt,
             spike_train=train,
             spike_costs=spike_costs,
             ca_cntrl=ca_cntrl,
             ca_letm1=ca_letm1)

if __name__ == '__main__':
    baseline_atp = 100
    firing_freq = 15
    t_start = 150  # ms
    t_run = 3200000  # ms
    time, tt, spike_vals, ca_cntrl, ca_letm1, sp_train = poi_spiking(sp_hz=firing_freq,
                                                                     q=10,
                                                                     t_start=t_start, t_run=t_run,
                                                                     t_ca_1=7000, t_ca_2=20000)
    rec_cntrl = spike_train_costs(time, tt, baseline_atp, spike_vals, ca_cntrl)
    rec_letm1 = spike_train_costs(time, tt, baseline_atp, spike_vals, ca_letm1)
    # rec_mcukd = spike_train_costs(time, tt, baseline_atp, spike_vals, ca_cntrl*0)

    labels = ['PDP1', 'Letm1+PDP1'] #, 'MCUKD']
    t_ca = [7, 20] #, 7]
    for ii, rec in enumerate([rec_cntrl, rec_letm1]): #, rec_mcukd]):
        filename = str(seed) + '_PDP_slower_' + str(baseline_atp) + '_poi_'+str(int(firing_freq))+'Hz_' + str(t_ca[ii]) + '_' + labels[ii] + '.npz'
        dump_vals(filename, rec, tt, sp_train, spike_vals, ca_cntrl, ca_letm1)
        

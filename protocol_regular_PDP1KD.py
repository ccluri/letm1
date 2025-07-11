import numpy as np
from scipy import stats
from utils import Recorder, Q_nak
from mitochondria_PDP1KD import Mito
import matplotlib.pyplot as plt
from figure_properties import *

dt = 0.01


def regular_spiking(sp_hz, spike_dur, q, t_start=150, t_run=1000, t_ca=5000, ca_fact=0.1):
    time = t_run + t_start + spike_dur  # ms total sim time
    tt = np.arange(0, time, dt)
    isi = 1000 / sp_hz  # in ms
    tot_spks = int(spike_dur/isi)
    print('Total spikes: ', tot_spks)
    spike_costs = np.zeros_like(tt)
    ca_matrix = np.zeros_like(tt)
    Q_val = Q_nak(tt, q, tau_Q=100)  # ATP expenditure increase per-spike
    ca_influx = Q_nak(tt, fact=ca_fact, tau_Q=t_ca)
    for ii in range(tot_spks):
        t_offset = ii*isi
        t_idx = int((t_start+t_offset)/dt)
        spike_costs[t_idx:] += Q_val[:len(spike_costs[t_idx:])]
        ca_matrix[t_idx:] += ca_influx[:len(ca_matrix[t_idx:])]
    return time, tt, spike_costs, ca_matrix


def spike_train_costs(time, tt, baseline_atp, spike_costs, ca_matrix, special=False):
    '''Perturbations due to a spike train'''
    factor = 0.1e-4  # PSI leak factor for Ca per-spike
    m = Mito(baseline_atp=baseline_atp)
    m.steadystate_vals(time=2000)  # steady state - wait till we reach here
    # rec_vars_list = ['atp', 'ca_mat', 'jjV', 'jjP', 'jjC', 'jjK', 'v1']
    if special:
        rec_vars_list = ['atp']
    else:
        rec_vars_list = ['atp', 'ca_mat']
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


def dump_vals(filename, rec, tt, special=False):
    if not special:
        np.savez(filename,
                 atp=rec.out['atp'],
                 ca_mat=rec.out['ca_mat'],
                 # v1=rec.out['v1'],
                 tt=tt,)
                 #jjV=rec.out['jjV'])
    else:
        np.savez(filename,
                 atp=rec.out['atp'],
                 )

if __name__ == '__main__':
    baseline_atp = 100
    MCU_KD = False
    
    firing_freq = 10
    time, tt, spike_vals, ca_pdp1 = regular_spiking(sp_hz=firing_freq, spike_dur=60000, q=10,
                                                    t_start=120000, t_run=480000, t_ca=7000)
    time, tt, spike_vals, ca_letm1_pdp1 = regular_spiking(sp_hz=firing_freq, spike_dur=60000, q=10,
                                                          t_start=120000, t_run=480000, t_ca=20000)
    if MCU_KD:
        time, tt, spike_vals, ca_mcukd = regular_spiking(sp_hz=firing_freq, spike_dur=60000, q=10,
                                                         t_start=10000, t_run=240000, t_ca=7000, ca_fact=0)

    # firing_freq = 20
    # time, tt, spike_vals, ca_cntrl = regular_spiking(sp_hz=firing_freq, spike_dur=1000, q=10,
    #                                                  t_start=1000, t_run=30000, t_ca=7000)
    # time, tt, spike_vals, ca_letm1 = regular_spiking(sp_hz=firing_freq, spike_dur=1000, q=10,
    #                                                  t_start=1000, t_run=30000, t_ca=20000)
    # if MCU_KD:
        # time, tt, spike_vals, ca_mcukd = regular_spiking(sp_hz=firing_freq, spike_dur=1000, q=10,
        #                                                  t_start=1000, t_run=30000, t_ca=7000, ca_fact=0)
    
    rec_pdp1 = spike_train_costs(time, tt, baseline_atp, spike_vals, ca_pdp1, special=True)
    rec_letm1_pdp1 = spike_train_costs(time, tt, baseline_atp, spike_vals, ca_letm1_pdp1, special=True)
    if MCU_KD:
        rec_mcukd = spike_train_costs(time, tt, baseline_atp, spike_vals, ca_mcukd)

    labels = ['PDP1', 'Letm1+PDP1']
    t_ca = [7, 20]
    recs = [rec_pdp1, rec_letm1_pdp1]
    if MCU_KD:
        labels.append('MCUKD')
        t_ca.append(7)
        recs.append(rec_mcudk)

    for ii, rec in enumerate(recs):
        filename = '100000tau_' + str(baseline_atp) + '_regular_'+str(int(firing_freq))+'Hz_' + str(t_ca[ii]) + '_' + labels[ii] + '_600AP_2long.npz'
        dump_vals(filename, rec, tt, special=True)
        
   
        
    # for ii, rec in enumerate(recs):
    #     filename = str(baseline_atp) + '_regular_'+str(int(firing_freq))+'Hz_' + str(t_ca[ii]) + '_' + labels[ii] + '.npz'
    #     dump_vals(filename, rec, tt)



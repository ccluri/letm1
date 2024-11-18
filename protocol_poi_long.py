import numpy as np
from scipy import stats
from utils import Recorder, Q_nak
from mitochondria import Mito
import matplotlib.pyplot as plt
from figure_properties import *


# This version was used to test the long simulation protocol for 2Hz (let it run after spikes)

dt = 0.01


def generate_spiketrain(rate, duration):  # duration given in ms
    n_spikes = int(duration*rate/1000)
    isi = stats.expon.rvs(scale=1./rate, size=n_spikes)
    spike_train = 1000*np.cumsum(isi) # returning also in ms
    spike_train = spike_train[np.where(spike_train < duration)]
    return spike_train


def poi_spiking(sp_hz, q, t_start=150, t_run=1000, t_ca_1=5000, t_ca_2=7000):
    time = t_run + t_start  # ms total sim time
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


def regular_spiking(sp_hz, spike_dur, q, t_start=150, t_run=1000, t_ca=5000):
    time = t_run + t_start + spike_dur  # ms total sim time
    tt = np.arange(0, time, dt)
    isi = 1000 / sp_hz  # in ms
    tot_spks = int(spike_dur/isi)
    print('Total spikes: ', tot_spks)
    spike_costs = np.zeros_like(tt)
    ca_matrix = np.zeros_like(tt)
    Q_val = Q_nak(tt, q, tau_Q=100)  # ATP expenditure increase per-spike
    ca_influx = Q_nak(tt, fact=0.5, tau_Q=t_ca)
    for ii in range(tot_spks):
        t_offset = ii*isi
        t_idx = int((t_start+t_offset)/dt)
        spike_costs[t_idx:] += Q_val[:len(spike_costs[t_idx:])]
        ca_matrix[t_idx:] += ca_influx[:len(ca_matrix[t_idx:])]
    return time, tt, spike_costs, ca_matrix


def spike_train_costs(time, tt, baseline_atp, spike_costs, ca_matrix):
    '''Perturbations due to a spike train'''
    factor = 0.1e-4  # PSI leak factor for Ca per-spike
    m = Mito(baseline_atp=baseline_atp)
    m.steadystate_vals(time=2000)  # steady state - wait till we reach here
    rec_vars_list = ['atp', 'ca_mat', 'jjV', 'v1']
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
             ca_mat=rec.out['ca_mat'],
             v1=rec.out['v1'],
             jjV=rec.out['jjV'],
             tt=tt,
             spike_train=train)

if __name__ == '__main__':
    baseline_atp = 100
    time, tt, spike_vals, ca_cntrl, ca_letm1, sp_train = poi_spiking(sp_hz=2, q=10,
                                                                     t_start=150, t_run=100000,
                                                                     t_ca_1=7000, t_ca_2=20000)
    rec_cntrl = spike_train_costs(time, tt, baseline_atp, spike_vals, ca_cntrl)
    rec_letm1 = spike_train_costs(time, tt, baseline_atp, spike_vals, ca_letm1)

    labels = ['Control', 'Letm1']
    t_ca = [7, 20]
    for ii, rec in enumerate([rec_cntrl, rec_letm1]):
        filename = str(baseline_atp) + '_poi_2Hz_long_' + str(t_ca[ii]) + '_' + labels[ii] + '.npz'
        dump_vals(filename, rec, tt, sp_train)
    

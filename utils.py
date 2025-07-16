import numpy as np


class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

class Recorder_select():
    def __init__(self, inst, rec_vars_list):
        self.inst = inst
        self.rec_vars_list = rec_vars_list
        self.out = {}
        for item in rec_vars_list:
            self.out[item] = []
        self.rec_times = []

    def update(self, tt):
        for item in self.rec_vars_list:
            try:
                rec = getattr(self.inst, item)
            except AttributeError:  # go deeper
                start = self.inst
                for sub_i in item.split('.'):
                    start = getattr(start, sub_i)
                rec = start
            self.out[item].append(rec)
        self.rec_times.append(tt)
        return

    def convert_to_arrays(self):
        for item in self.rec_vars_list:
            self.out[item] = np.array(self.out[item])
        self.rec_times = np.array(self.rec_times)
        return
            

        
class Recorder():
    def __init__(self, inst, rec_vars_list, time, dt):
        self.inst = inst
        self.time = time
        self.dt = dt
        self.out = {}
        for item in rec_vars_list:
            self.out[item] = np.zeros(int(time/dt))
        self.rec_vars_list = rec_vars_list

    def update(self, ii):
        for item in self.rec_vars_list:
            try:
                self.out[item][ii] = getattr(self.inst, item)
            except AttributeError:  # go deeper
                start = self.inst
                for sub_i in item.split('.'):
                    start = getattr(start, sub_i)
                self.out[item][ii] = start
        return

    def adjust_var(self, item, spike_times, preset):
        tt = np.arange(0, self.time, self.dt)
        idxs = np.where(np.in1d(tt, np.array(spike_times)))[0]
        new_vals = np.array(self.out[item])
        new_vals[idxs] = preset
        self.out[item] = list(new_vals)


def Q_nak(tt, fact=1, tau_Q=100, tau_rise=0.6):
    vals = 1/(1*np.exp((tt-1)/tau_Q) + 30*np.exp(-(tt-3)/tau_rise))
    nrm = np.max(vals)
    return fact*vals/nrm



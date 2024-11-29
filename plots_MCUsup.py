import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import figure_properties as fp
import numpy as np

seed = 11

# 20 Hz, 20AP
c_reg_fn = '100_regular_20Hz_7_Control.npz'
l_reg_fn = '100_regular_20Hz_7_MCUKD.npz'

# 10 Hz, 10AP
c_reg_10 = '100_regular_10Hz_7_Control.npz'
l_reg_10 = '100_regular_10Hz_7_MCUKD.npz'

# 600 AP, 10Hz
c_reg_long = '100_regular_10Hz_7_Control_600AP.npz'
l_reg_long = '100_regular_10Hz_7_MCUKD_600AP.npz'

c1_poi_fn = str(seed) + '_100_poi_1Hz_7_Control.npz'
l1_poi_fn = str(seed) + '_100_poi_1Hz_7_MCUKD.npz'

c2_poi_fn = str(seed) + '_100_poi_2Hz_7_Control.npz'
l2_poi_fn = str(seed) + '_100_poi_2Hz_7_MCUKD.npz'

c5_poi_fn = str(seed) + '_100_poi_5Hz_7_Control.npz'
l5_poi_fn = str(seed) + '_100_poi_5Hz_7_MCUKD.npz'

c10_poi_fn = str(seed) + '_100_poi_10Hz_7_Control.npz'
l10_poi_fn = str(seed) + '_100_poi_10Hz_7_MCUKD.npz'

c20_poi_fn = str(seed) + '_100_poi_20Hz_7_Control.npz'
l20_poi_fn = str(seed) + '_100_poi_20Hz_7_MCUKD.npz'

c50_poi_fn = str(seed) + '_100_poi_50Hz_7_Control.npz'
l50_poi_fn = str(seed) + '_100_poi_50Hz_7_MCUKD.npz'


c_reg = np.load(c_reg_fn)
l_reg = np.load(l_reg_fn)
c_reg_t = np.load(c_reg_10)
l_reg_t = np.load(l_reg_10)
c_reg_l = np.load(c_reg_long)
l_reg_l = np.load(l_reg_long)

c1_poi = np.load(c1_poi_fn)
l1_poi = np.load(l1_poi_fn)
c2_poi = np.load(c2_poi_fn)
l2_poi = np.load(l2_poi_fn)
c5_poi = np.load(c5_poi_fn)
l5_poi = np.load(l5_poi_fn)
c10_poi = np.load(c10_poi_fn)
l10_poi = np.load(l10_poi_fn)
c20_poi = np.load(c20_poi_fn)
l20_poi = np.load(l20_poi_fn)
c50_poi = np.load(c50_poi_fn)
l50_poi = np.load(l50_poi_fn)

layout = [['.', 'A', '.', 'B'],
          ['S', 'S', 'S', 'S'],
          ['X', 'X', 'Y', 'Y'],
          ['P', 'P', 'Q', 'Q'],
          ['R', 'R', 'T', 'T'],
          ['U', 'U', 'V', 'V'],]

fig, axd = plt.subplot_mosaic(layout, figsize=fp.cm_to_inches([15, 17]),
                              height_ratios=[1, 0.25, 1, 1, 1, 1],
                              width_ratios=[1, 1, 1, 1])

# Regular firing
axd['A'].plot(c_reg['tt'], c_reg['ca_mat'], c='k', lw=1)
axd['A'].plot(l_reg['tt'], l_reg['ca_mat'], c='#00bfff', lw=1)
axd['A'].plot([1000, 2000], [2.2]*2, lw=2, c='k')
axd['A'].text(1500, 2.4, s='20AP, 20Hz', ha='center', va='center', fontsize=7)
axd['A'].set_ylim(-1.5, 2.5)
ymin, ymax = axd['A'].get_ylim()
ww = (ymax-ymin)/100
fp.remove_ticks_labels([axd['A']])
axd['A'].set_ylabel('Ca (mito) conc.')
fp.add_sizebar(axd['A'], 10000, size_vertical=ww, text='10 sec')

# # 600AP, 10Hz
# axd['B'].plot(c_reg_l['tt'], c_reg_l['atp'], c='k', lw=1)
# axd['B'].plot(l_reg_l['tt'], l_reg_l['atp'], c='#00bfff', lw=1)
# axd['B'].plot([10000, 70000], [0.9]*2, lw=2, c='k')
# axd['B'].text(40000, .95, s='600AP, 10Hz', ha='center', va='center', fontsize=7)
# axd['B'].set_ylim(0.3, 0.95)
# ymin, ymax = axd['B'].get_ylim()
# ww = (ymax-ymin)/100
# fp.remove_ticks_labels([axd['B']])
# axd['B'].set_ylabel('ATP conc.')
# fp.add_sizebar(axd['B'], 60000, size_vertical=ww, text='1 min')
# # axd['B'].set_title('Regular spikes, 600AP 10Hz')

# 20AP 20 Hz
axd['B'].plot(c_reg['tt'], c_reg['atp'], c='k', lw=1)
axd['B'].plot(l_reg['tt'], l_reg['atp'], c='#00bfff', lw=1)
axd['B'].plot([1000, 2000], [0.6]*2, lw=2, c='k')
axd['B'].text(1500, 0.62, s='20AP, 20Hz', ha='center', va='center', fontsize=7)
axd['B'].set_ylim(0.45, 0.65)
ymin, ymax = axd['B'].get_ylim()
ww = (ymax-ymin)/100
fp.remove_ticks_labels([axd['B']])
axd['B'].set_ylabel('ATP conc.')
fp.add_sizebar(axd['B'], 10000, size_vertical=ww, text='10 sec')
# axd['B'].set_title('Regular spikes, 20AP 20Hz')

# # 10AP 10 Hz
# axd['B'].plot(c_reg_t['tt'], c_reg_t['atp'], c='k', lw=1)
# axd['B'].plot(l_reg_t['tt'], l_reg_t['atp'], c='#00bfff', lw=1)
# axd['B'].plot([1000, 2000], [0.57]*2, lw=2, c='k')
# axd['B'].text(1500, 0.58, s='10AP, 10Hz', ha='center', va='center', fontsize=7)
# #axd['B'].set_ylim(0.45, 0.5)
# ymin, ymax = axd['B'].get_ylim()
# ww = (ymax-ymin)/100
# fp.remove_ticks_labels([axd['B']])
# axd['B'].set_ylabel('ATP conc.')
# fp.add_sizebar(axd['B'], 10000, size_vertical=ww, text='10 sec')
# #axd['B'].set_title('Regular spikes, 10AP 10Hz')


# poisson spiking
total_sample = len(c2_poi['tt'])
spike_sample = total_sample / 2
first_ten_idx = len(np.arange(0, 10*1000, 0.01))  # first 10 secs
tot_spks = len(c2_poi['spike_train'])

last_spk_idx_1 = int(c1_poi['spike_train'][-1]*100)
first_spk_idx_1 = int(c1_poi['spike_train'][0]*100)
last_spk_idx_2 = int(c2_poi['spike_train'][-1]*100)
first_spk_idx_2 = int(c2_poi['spike_train'][0]*100)
last_spk_idx_5 = int(c5_poi['spike_train'][-1]*100)
first_spk_idx_5 = int(c5_poi['spike_train'][0]*100)
last_spk_idx_10 = int(c10_poi['spike_train'][-1]*100)
first_spk_idx_10 = int(c10_poi['spike_train'][0]*100)
last_spk_idx_20 = int(c20_poi['spike_train'][-1]*100)
first_spk_idx_20 = int(c20_poi['spike_train'][0]*100)
last_spk_idx_50 = int(c50_poi['spike_train'][-1]*100)
first_spk_idx_50 = int(c50_poi['spike_train'][0]*100)

axd['S'].vlines(c2_poi['spike_train'], 0.2, 0.8, color='gray', lw=0.4)
# axd['S'].set_xlim(-1000, c2_poi['tt'][-1])  # last_spk_idx/100)
axd['S'].set_xlim(-1000, 1.02*(last_spk_idx_2/100))

l_inset = Rectangle((-0.1, -0.1), height=1.2, width=first_ten_idx/100,
                    transform=axd['S'].transData, fill=False)
axd['S'].add_patch(l_inset)
r_inset = Rectangle((-0.1+(last_spk_idx_2-first_ten_idx)/100, -0.1), height=1.2,
                    width=1*first_ten_idx/100,
                    transform=axd['S'].transData, fill=False)
axd['S'].add_patch(r_inset)

# axd['S'].annotate("", xy=(0, 1.6), xytext=((last_spk_idx-first_spk_idx)/100, 1.6),
#                   textcoords=axd['S'].transData, arrowprops=dict(arrowstyle='<->'))
# axd['S'].annotate("", xy=(0, 1.6), xytext=((last_spk_idx-first_spk_idx)/100, 1.6),
#                   textcoords=axd['S'].transData, arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5'))
bbox = dict(fc="white", ec="none")
axd['S'].text((last_spk_idx_2-first_spk_idx_2)/200, 1.6,
              s='Random %s AP, 100 sec (~2Hz)' % str(tot_spks),
              ha='center', va='center', fontsize=7, clip_on=False, bbox=bbox)
# axd['S'].annotate("", xy=(last_spk_idx/100, 1.6),
#                   xytext=(c2_poi['tt'][-1], 1.6),
#                   textcoords=axd['S'].transData, arrowprops=dict(arrowstyle='<->'))
# axd['S'].annotate("", xy=(last_spk_idx/100, 1.6),
#                   xytext=(c2_poi['tt'][-1], 1.6),
#                   textcoords=axd['S'].transData, arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5'))
# bbox = dict(fc="white", ec="none")
# axd['S'].text(c2_poi['tt'][-1]-(last_spk_idx)/200, 1.6,
#               s='virtual TTX / no activity',
#               ha='center', va='center', fontsize=7, clip_on=False, bbox=bbox)

axd['S'].set_ylabel('Spikes', labelpad=13)
axd['S'].set_ylim(-0.2, 1.2)

con = ConnectionPatch(xyA=(-0.1, -0.1), coordsA=axd['S'].transData,
                      xyB=(0, 1), coordsB=axd['X'].transAxes)
fig.add_artist(con)
con = ConnectionPatch(xyA=(-0.1+(first_ten_idx/100), -0.1),
                      coordsA=axd['S'].transData,
                      xyB=(1, 1), coordsB=axd['X'].transAxes)
fig.add_artist(con)

con = ConnectionPatch(xyA=(-0.1+(last_spk_idx_2-first_ten_idx)/100, -0.1),
                      coordsA=axd['S'].transData,
                      xyB=(0, 1), coordsB=axd['Y'].transAxes)
fig.add_artist(con)
con = ConnectionPatch(xyA=((last_spk_idx_2)/100, -0.1),
                      coordsA=axd['S'].transData,
                      xyB=(1, 1), coordsB=axd['Y'].transAxes)
fig.add_artist(con)


axd['X'].vlines(c2_poi['spike_train'], 0, 1, color='gray', lw=0.4)
axd['X'].plot(c2_poi['tt'], c2_poi['atp'], c='k', lw=1)
axd['X'].plot(l2_poi['tt'], l2_poi['atp'], c='#00bfff', lw=1)
axd['X'].set_ylim(0.45, 0.65)
axd['X'].set_xlim(0, first_ten_idx/100)
axd['X'].set_ylabel('ATP conc. (A.U.)', labelpad=13)

axd['Y'].vlines(c2_poi['spike_train'], 0, 1, color='gray', lw=0.4)
axd['Y'].plot(c2_poi['tt'], c2_poi['atp'], c='k', lw=1)
axd['Y'].plot(l2_poi['tt'], l2_poi['atp'], c='#00bfff', lw=1)
axd['Y'].set_ylim(0.45, 0.65)
axd['Y'].set_xlim((last_spk_idx_2-first_ten_idx)/100,
                  (last_spk_idx_2)/100)

a0 = c1_poi['atp'][: first_spk_idx_1]
b0 = l1_poi['atp'][: first_spk_idx_1]
a1 = c1_poi['atp'][last_spk_idx_1-first_ten_idx:last_spk_idx_1]
b = l1_poi['atp'][last_spk_idx_1-first_ten_idx:last_spk_idx_1]
a = c2_poi['atp'][last_spk_idx_2-first_ten_idx:last_spk_idx_2]
c = l2_poi['atp'][last_spk_idx_2-first_ten_idx:last_spk_idx_2]
d5 = c5_poi['atp'][last_spk_idx_5-first_ten_idx:last_spk_idx_5]
d = l5_poi['atp'][last_spk_idx_5-first_ten_idx:last_spk_idx_5]
# ax = c2_poi['atp'][-first_ten_idx:]
# cx = l2_poi['atp'][-first_ten_idx:]
axd['P'].bar(x=[-0.125, 0.125, 0.875, 1.125, 1.875, 2.125, 2.875, 3.125],
             height=[np.mean(ii) for ii in [a0, b0, a1, b, a, c, d5, d]],
             color=['k', '#b5b5b5',
                    'k', 'paleturquoise',
                    'k', '#00bfff',
                    'k', 'blue'], width=0.25)
axd['P'].set_xticks([0, 1, 2, 3])
axd['P'].set_xticklabels(['0Hz', '1Hz', '2Hz', '5Hz'])
axd['P'].set_ylim(0, 1)
axd['P'].set_yticks([0, 0.5, 1])
axd['P'].set_ylabel('ATP conc. (A.U.)')
axd['P'].set_title('10s Mean ATP (during spikes)')

a0 = c1_poi['pyr'][:first_spk_idx_1]
b0 = l1_poi['pyr'][:first_spk_idx_1]
a1 = c1_poi['pyr'][last_spk_idx_1-first_ten_idx:last_spk_idx_1]
b = l1_poi['pyr'][last_spk_idx_1-first_ten_idx:last_spk_idx_1]
a = c2_poi['pyr'][last_spk_idx_2-first_ten_idx:last_spk_idx_2]
c = l2_poi['pyr'][last_spk_idx_2-first_ten_idx:last_spk_idx_2]
d5 = c5_poi['pyr'][last_spk_idx_5-first_ten_idx:last_spk_idx_5]
d = l5_poi['pyr'][last_spk_idx_5-first_ten_idx:last_spk_idx_5]
axd['Q'].bar(x=[-0.125, 0.125, 0.875, 1.125, 1.875, 2.125, 2.875, 3.125],
             height=[np.mean(ii) for ii in [a0, b0, a1, b, a, c, d5, d]],
             color=['k', '#b5b5b5',
                    'k', 'paleturquoise',
                    'k', '#00bfff',
                    'k', 'blue'], width=0.25)
axd['Q'].set_xticks([0, 1, 2, 3])
axd['Q'].set_xticklabels(['0Hz', '1Hz', '2Hz', '5Hz'])
axd['Q'].set_ylim(0, 1.25)
axd['Q'].set_yticks([0, 0.5, 1])
axd['Q'].set_ylabel('Pyruvate (A.U.)')
axd['Q'].set_title('10s Mean Pyruvate (during spikes)')

vals_c = []
vals_l = []
a0 = c1_poi['atp'][:first_spk_idx_1]
b0 = l1_poi['atp'][:first_spk_idx_1]
vals_c.append(np.mean(a0))
vals_l.append(np.mean(b0))
c_poi = [c1_poi, c2_poi, c5_poi, c10_poi, c20_poi, c50_poi]
l_poi = [l1_poi, l2_poi, l5_poi, l10_poi, l20_poi, l50_poi]
l_spike_idx = [last_spk_idx_1, last_spk_idx_2, last_spk_idx_5,
               last_spk_idx_10, last_spk_idx_20, last_spk_idx_50]
for xx, yy, zz in zip(c_poi, l_poi, l_spike_idx):
    vals_c.append(np.mean(xx['atp'][zz-first_ten_idx:zz]))
    vals_l.append(np.mean(yy['atp'][zz-first_ten_idx:zz]))
freqs = [0, 1, 2, 5, 10, 20, 50]
print(vals_c, 'control atp during')
print(vals_l, 'mcukd atp during')
axd['R'].plot(freqs, vals_c, c='k', marker='.', label='Control')
axd['R'].plot(freqs, vals_l, c='#00bfff', marker='.', label='MCU KD')
axd['R'].set_ylabel('ATP conc. (A.U.)')
axd['R'].set_xlabel('Frequency (Hz)')
axd['R'].legend(frameon=False, ncols=2, loc='lower center')
axd['R'].set_ylim(0, 1.)
axd['R'].set_yticks([0, 0.5, 1])
axd['R'].set_title('10s Mean ATP (during spikes)')

vals_c = []
vals_l = []
a0 = c1_poi['pyr'][:first_spk_idx_1]
b0 = l1_poi['pyr'][:first_spk_idx_1]
vals_c.append(np.mean(a0))
vals_l.append(np.mean(b0))
c_poi = [c1_poi, c2_poi, c5_poi, c10_poi, c20_poi, c50_poi]
l_poi = [l1_poi, l2_poi, l5_poi, l10_poi, l20_poi, l50_poi]
l_spike_idx = [last_spk_idx_1, last_spk_idx_2, last_spk_idx_5,
               last_spk_idx_10, last_spk_idx_20, last_spk_idx_50]
for xx, yy, zz in zip(c_poi, l_poi, l_spike_idx):
    vals_c.append(np.mean(xx['pyr'][zz-first_ten_idx:zz]))
    vals_l.append(np.mean(yy['pyr'][zz-first_ten_idx:zz]))
freqs = [0, 1, 2, 5, 10, 20, 50]
print(vals_c, 'control pyruvate during')
print(vals_l, 'mcukd pyruvate during')
axd['T'].plot(freqs, vals_c, c='k', marker='.', label='Control')
axd['T'].plot(freqs, vals_l, c='#00bfff', marker='.', label='MCU KD')
axd['T'].set_ylabel('Pyruvate (A.U.)')
axd['T'].set_xlabel('Frequency (Hz)')
axd['T'].legend(ncols=1, frameon=False, loc='lower left')
axd['T'].set_ylim(0, 1.25)
axd['T'].set_yticks([0, 0.5, 1])
axd['T'].set_title('10s Mean Pyruvate (during spikes)')

vals_c = []
vals_l = []
a0 = c1_poi['ca_mat'][:first_spk_idx_1]
b0 = l1_poi['ca_mat'][:first_spk_idx_1]
vals_c.append(np.mean(a0))
vals_l.append(np.mean(b0))
c_poi = [c1_poi, c2_poi, c5_poi, c10_poi, c20_poi, c50_poi]
l_poi = [l1_poi, l2_poi, l5_poi, l10_poi, l20_poi, l50_poi]
l_spike_idx = [last_spk_idx_1, last_spk_idx_2, last_spk_idx_5,
               last_spk_idx_10, last_spk_idx_20, last_spk_idx_50]
for xx, yy, zz in zip(c_poi, l_poi, l_spike_idx):
    vals_c.append(np.mean(xx['ca_mat'][zz-first_ten_idx:zz]))
    vals_l.append(np.mean(yy['ca_mat'][zz-first_ten_idx:zz]))
freqs = [0, 1, 2, 5, 10, 20, 50]
print(vals_c, 'control ca_mat during')
print(vals_l, 'mcukd ca_mat during')


vals_c = []
vals_l = []
a0 = c1_poi['atp'][:first_spk_idx_1]
b0 = l1_poi['atp'][:first_spk_idx_1]
vals_c.append(np.mean(a0))
vals_l.append(np.mean(b0))
c_poi = [c1_poi, c2_poi, c5_poi, c10_poi, c20_poi, c50_poi]
l_poi = [l1_poi, l2_poi, l5_poi, l10_poi, l20_poi, l50_poi]
l_spike_idx = [last_spk_idx_1, last_spk_idx_2, last_spk_idx_5,
               last_spk_idx_10, last_spk_idx_20, last_spk_idx_50]
for xx, yy, zz in zip(c_poi, l_poi, l_spike_idx):
    vals_c.append(np.mean(xx['atp'][zz:zz+first_ten_idx]))
    vals_l.append(np.mean(yy['atp'][zz:zz+first_ten_idx]))
freqs = [0, 1, 2, 5, 10, 20, 50]
print(vals_c, 'control atp after')
print(vals_l, 'mcukd atp after')
axd['U'].plot(freqs, vals_c, c='k', marker='.', label='Control')
axd['U'].plot(freqs, vals_l, c='#00bfff', marker='.', label='MCU KD')
axd['U'].set_ylabel('ATP conc. (A.U.)')
axd['U'].set_xlabel('Frequency (Hz)')
axd['U'].legend(frameon=False, ncols=2, loc='lower center')
axd['U'].set_ylim(0, 1.)
axd['U'].set_yticks([0, 0.5, 1])
axd['U'].set_title('10s Mean ATP (after spikes)')

vals_c = []
vals_l = []
a0 = c1_poi['pyr'][:first_spk_idx_1]
b0 = l1_poi['pyr'][:first_spk_idx_1]
vals_c.append(np.mean(a0))
vals_l.append(np.mean(b0))
c_poi = [c1_poi, c2_poi, c5_poi, c10_poi, c20_poi, c50_poi]
l_poi = [l1_poi, l2_poi, l5_poi, l10_poi, l20_poi, l50_poi]
l_spike_idx = [last_spk_idx_1, last_spk_idx_2, last_spk_idx_5,
               last_spk_idx_10, last_spk_idx_20, last_spk_idx_50]
for xx, yy, zz in zip(c_poi, l_poi, l_spike_idx):
    vals_c.append(np.mean(xx['pyr'][zz:first_ten_idx+zz]))
    vals_l.append(np.mean(yy['pyr'][zz:first_ten_idx+zz]))
freqs = [0, 1, 2, 5, 10, 20, 50]
print(vals_c, 'control pyruvate after')
print(vals_l, 'mcukd pyruvate after')
axd['V'].plot(freqs, vals_c, c='k', marker='.', label='Control')
axd['V'].plot(freqs, vals_l, c='#00bfff', marker='.', label='MCU KD')
axd['V'].set_ylabel('Pyruvate (A.U.)')
axd['V'].set_xlabel('Frequency (Hz)')
axd['V'].legend(ncols=1, frameon=False, loc='lower left')
axd['V'].set_ylim(0, 1.25)
axd['V'].set_yticks([0, 0.5, 1])
axd['V'].set_title('10s Mean Pyruvate (after spikes)')

vals_c = []
vals_l = []
a0 = c1_poi['ca_mat'][:first_spk_idx_1]
b0 = l1_poi['ca_mat'][:first_spk_idx_1]
vals_c.append(np.mean(a0))
vals_l.append(np.mean(b0))
c_poi = [c1_poi, c2_poi, c5_poi, c10_poi, c20_poi, c50_poi]
l_poi = [l1_poi, l2_poi, l5_poi, l10_poi, l20_poi, l50_poi]
l_spike_idx = [last_spk_idx_1, last_spk_idx_2, last_spk_idx_5,
               last_spk_idx_10, last_spk_idx_20, last_spk_idx_50]
for xx, yy, zz in zip(c_poi, l_poi, l_spike_idx):
    vals_c.append(np.mean(xx['ca_mat'][zz:zz+first_ten_idx]))
    vals_l.append(np.mean(yy['ca_mat'][zz:zz+first_ten_idx]))
freqs = [0, 1, 2, 5, 10, 20, 50]
print(vals_c, 'control ca_mat after')
print(vals_l, 'mcukd ca_mat after')

for quant in ['ca_mat', 'pyr', 'atp']:
    vals_c = []
    vals_l = []
    a0 = c1_poi[quant][first_spk_idx_1]
    b0 = l1_poi[quant][first_spk_idx_1]
    vals_c.append(a0)
    vals_l.append(b0)
    c_poi = [c1_poi, c2_poi, c5_poi, c10_poi, c20_poi, c50_poi]
    l_poi = [l1_poi, l2_poi, l5_poi, l10_poi, l20_poi, l50_poi]
    l_spike_idx = [last_spk_idx_1, last_spk_idx_2, last_spk_idx_5,
                   last_spk_idx_10, last_spk_idx_20, last_spk_idx_50]
    for xx, yy, zz in zip(c_poi, l_poi, l_spike_idx):
        vals_c.append(xx[quant][zz+100000])
        vals_l.append(yy[quant][zz+100000])
    print(vals_c, 'control 101 '+quant)
    print(vals_l, 'mcukd 101 '+quant)

aa = ['A', 'B', 'S', 'X', 'Y']
fp.remove_spines([axd[ii] for ii in aa], bottom=True, left=True)
fp.remove_spines([axd['P'], axd['Q'], axd['R'], axd['T'], axd['U'], axd['V']])
fp.remove_ticks_labels([axd['S'], axd['X'], axd['Y']])

plt.tight_layout()
plt.savefig('20Hz_MCUKD.svg')
#plt.show()

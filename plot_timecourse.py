import numpy as np
import matplotlib.pyplot as plt



def load_file(fname):
    data = np.load(fname)
    dt = 0.1
    times = data['times']
    idx = np.where(np.diff(times) > 1)[0]
    idx_end = np.append(idx, len(times))
    idx_strt = np.append(0, idx+1)
    atps = []
    pyrs = []
    print('At time 0, ATP:', data['atp'][0])
    print('At time 0, pyr:', data['pyr'][0])
    for ii in range(len(idx_strt)):
        atps.append(np.mean(data['atp'][idx_strt[ii]:idx_end[ii]]))
        pyrs.append(np.mean(data['pyr'][idx_strt[ii]:idx_end[ii]]))
        # atps.append(np.median(data['atp'][idx_strt[ii]:idx_end[ii]]))
        # pyrs.append(np.median(data['pyr'][idx_strt[ii]:idx_end[ii]]))
    return atps, pyrs


#all_atps = [[0.5347688947977433], [0.5347688947977433], [0.5347688947977433], [0.5347688947977433]]  # value at t=0
#all_pyrs = [[1.1492634632717318], [1.1492634632717318], [1.1492634632717318], [1.1492634632717318]]  # value at t=0

all_atps = [[], [], [], []]
all_pyrs = [[], [], [], []]

seed = 11
freq = [1, 2, 5, 10, 15]
start_rec_at = [0, 90, 190, 390, 790, 1590, 3190] # SEC

labels = ['Control', 'Letm1', 'PDP1', 'PDP1+Letm1']
colors = ['k', 'C0', 'C1', 'C2']

for ff in freq:
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    c_poi = str(seed) + '_slower_100_poi_' + str(ff) + 'Hz_7_Control.npz'
    l_poi = str(seed) + '_slower_100_poi_' + str(ff) + 'Hz_20_Letm1.npz'
    p_poi = str(seed) + '_PDP_slower_100_poi_' + str(ff) + 'Hz_7_PDP1.npz'
    pl_poi = str(seed) + '_PDP_slower_100_poi_' + str(ff) + 'Hz_20_Letm1+PDP1.npz'
    all_fnames = [c_poi, l_poi, p_poi, pl_poi] 
    for ii, (cc, label, fname) in enumerate(zip(colors, labels, all_fnames)):
    #ff = 15
        atps, pyrs = load_file(fname)
        ax.plot(start_rec_at, atps, c=cc, marker='o', label=label)
        ax2.plot(start_rec_at, pyrs, c=cc, marker='o', label=label)
        all_atps[ii].append(atps)
        all_pyrs[ii].append(pyrs)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('ATP (a.u.)')
    #ax.set_title('Random spikes @ ' + str(ff) + ' Hz')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Pyruvate (a.u.)')
    #ax2.set_title('Random spikes @ ' + str(ff) + ' Hz')

    ax.set_ylim(0.4, 0.8)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xticks(start_rec_at)
    ax.set_xticklabels([ii+10 for ii in start_rec_at])
    ax.legend(frameon=False, loc=9, ncols=2)

    ax2.set_ylim(0.6, 1.4)
    ax2.set_yticks([0.6, 0.8, 1, 1.2])
    ax2.set_xticks(start_rec_at)
    ax2.set_xticklabels([ii+10 for ii in start_rec_at])

    ax.spines[['right', 'top']].set_visible(False)
    ax2.spines[['right', 'top']].set_visible(False)
    fig.autofmt_xdate()
    plt.suptitle('Random spikes @ ' + str(ff) + ' Hz')
    plt.tight_layout()
    plt.savefig(str(seed)+'_' +str(ff)+ '_ATP+Pyr_vs_time.png')
    plt.close(fig)
    
atp_0 = 0.5347688947977433
pyr_0 = 1.1492634632717318
    
import csv
for ii, fname in enumerate(labels):
    with open(fname + '_atp.csv', 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(['Time(s)->', 0] + [jj+10 for jj in start_rec_at])
        csvwriter.writerow([0]+[atp_0]*(len(start_rec_at)+1))  # 0 freq
        for ff, row in enumerate(all_atps[ii]):
            csvwriter.writerow([freq[ff], atp_0] + row)
    with open(fname + '_pyr.csv', 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(['Time(s)->', 0] + [jj+10 for jj in start_rec_at])
        csvwriter.writerow([0]+[pyr_0]*(len(start_rec_at)+1))  # 0 freq
        for ff, row in enumerate(all_pyrs[ii]):
            csvwriter.writerow([freq[ff], pyr_0] + row)

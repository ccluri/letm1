import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    """
    Calculates the moving average of a 1D array.

    Args:
        data: A 1D numpy array or list of numerical data.
        window_size: The size of the moving window (must be a positive integer).

    Returns:
        A 1D numpy array containing the moving averages.  Returns an empty array if 
        window_size is not a positive integer or if the input data is empty. Returns
        the original data if window_size is 1.

    Raises:
        TypeError: If data is not a list or numpy array.
        ValueError: If window_size is not a positive integer.
    """

    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("Input data must be a list or numpy array.")

    if not isinstance(window_size, int):
        raise TypeError("Window size must be an integer.")

    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
    
    
    if window_size == 1:
        return np.array(data) if isinstance(data, list) else data.copy() # Return copy to avoid modifying original

    data = np.array(data)  # Convert to numpy array for efficiency

    if window_size > len(data): # Handle cases where window is larger than data
        return np.array([]) # Or you might choose to return the mean of the entire data set

    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size


# all in ms
dt = 0.01
spike_dur = 60000
t_start = 120000
t_run = 480000
total = spike_dur + t_start + t_run

tt = np.arange(0, total, dt)

ms_to_sec = lambda x : x / 100
sec_to_min = lambda x : x / 60
window = 60 # second
window_size = 60*1000*int(1/0.01)
min_window_size = 6000000 # 1000*int(1/0.01)
total_min = sec_to_min(ms_to_sec(total))

ll = ['Control', 'Letm1', 'PDP1', 'PDP1+Letm1']
cs = ['k', 'C0', 'C1', 'C2']

for jj, ii in enumerate([
        '100000tau_100_regular_10Hz_7_Control_600AP_2long.npz',
        '100000tau_100_regular_10Hz_20_Letm1_600AP_2long.npz',
        '100_regular_10Hz_7_PDP1_600AP_2long.npz',
        '100_regular_10Hz_20_Letm1+PDP1_600AP_2long.npz']):
    data = np.load(ii)
    vv = [data['atp'][0]]
    tpts = [0]
    for tt in range(1, 11):
        vv.append(np.mean(data['atp'][(tt-1)*min_window_size: (tt*min_window_size)-1]))
        tpts.append(tt)
        print(vv[-1], tt)
    plt.plot(tpts, vv, label=ll[jj], c=cs[jj], lw=2)
    # if jj == 0:
    #     zscore=10
    # else:
    #     zscore=9
    # plt.plot(data['atp'][::5000], label=ll[jj], c=cs[jj], zorder=zscore, lw=0.5, alpha=0.8)
plt.legend(frameon=False)
plt.xlabel('Time (mins)')
plt.ylabel('ATP (a.u.)')

# plt.xticks([0, 2000, 4000, 6000, 8000, 10000, 12000], [0, 2, 4, 6, 8, 10, 12])

plt.savefig('100000tau_compare_all.png')
plt.show()

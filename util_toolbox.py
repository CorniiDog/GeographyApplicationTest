import numpy as np

def text_box_plot(data, width=50):
    data = np.array(data)
    data = data[~np.isnan(data)]
    if data.size == 0:
        print("No data to plot.")
        return

    mn = np.min(data)
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    mx = np.max(data)

    # Define a helper to map a data value to a position on a scale of given width.
    def pos(value):
        if mx == mn:
            return 0
        return int((value - mn) / (mx - mn) * width)
    
    p_min = pos(mn)
    p_q1 = pos(q1)
    p_med = pos(med)
    p_q3 = pos(q3)
    p_max = pos(mx)

    # Build the scale line
    line = [' '] * (width + 1)
    line[p_min] = '|'
    line[p_q1] = '['
    line[p_med] = '|'
    line[p_q3] = ']'
    line[p_max] = '|'
    plot_line = ''.join(line)

    print("Min: {:.2f}   Q1: {:.2f}   Median: {:.2f}   Q3: {:.2f}   Max: {:.2f}".format(mn, q1, med, q3, mx))
    print(plot_line)
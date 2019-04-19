STEPSZ = 'stepsz'


def create_grid(imsz, stepsz, cellsz, scales):
    h0, w0 = imsz
    h_step, w_step = stepsz
    h_cell, w_cell = cellsz
    l = []
    for s in scales:
        hs = int(h0 * s)
        ws = int(w0 * s)
        ls = []
        grid_y, grid_x = (int((hs - h_cell) / h_step) + 1, int((ws - w_cell) / w_step) + 1)
        for y in range(grid_y):
            y_start = y * h_step
            for x in range(grid_x):
                x_start = x * w_step
                ls.append((y_start, x_start))
        l.append((ls, s, hs, ws, grid_y, grid_x))

    return {'grid': l, 'imsz': imsz, ('%s' % STEPSZ): stepsz, 'cellsz': cellsz, 'scales': scales}


import matplotlib.pyplot as plt
import numpy as np

def show_fe(data, size, interpolate=None, cmap="jet"):
    xmin = np.min(data[:, 0])
    xmax = np.max(data[:, 0])
    ymin = np.min(data[:, 1])
    ymax = np.max(data[:, 1])
    xm = (xmax - xmin) * 0.1
    ym = (ymax - ymin) * 0.1
    xmin -= xm
    xmax += xm
    ymin -= ym
    ymax += ym
    x = np.linspace(xmin, xmax, size)
    y = np.linspace(ymin, ymax, size)
    count = np.zeros((size, size))
    for i in range(data.shape[0]):
        count[np.argmin(np.abs(x.T - data[i, 0])),
        np.argmin(np.abs(y.T - data[i, 1]))] += 1
    img = -np.log(count / count.max())
    img[img == np.inf] = img[img != np.inf].max()

    fig, ax = plt.subplots(1,1)
    if interpolate is not None:
        im = ax.imshow(img.T[::-1, :],
                       cmap=cmap, interpolation=interpolate,
                       extent=[xmin, xmax, ymin, ymax])
    else:
        xx, yy = np.mgrid[xmin:xmax:size * 1j, ymin:ymax:size * 1j]
        im = ax.contourf(xx, yy, img, cmap=cmap, levels=12)
    #
    cbar = fig.colorbar(im)
    cbar.set_label("$\Delta G / k_{B}T$")
    fig.show()


def viewAFM(img, vsize=1.0):
    if isinstance(img , list):
        nimg = len(img)
    else:
        nimg = 1
        img = [img]
    max_amp = max([img[i].max() for i in range(nimg)])

    fig, ax = plt.subplots(1, nimg, figsize = (5*nimg,4))
    if nimg == 1:
        extent = vsize * img[0].shape[0]//2
        im = ax.imshow(img[0].T, origin="lower", cmap="afmhot", vmax = max_amp , aspect='auto',
                       extent=[-extent,extent,-extent,extent])
        ax.grid(False)
        cbar = fig.colorbar(im)

    else:
        for i in range(nimg):
            extent = vsize * img[i].shape[0] // 2
            im = ax[i].imshow(img[i].T, origin="lower", cmap="afmhot", vmax = max_amp, aspect='auto',
                              extent=[-extent,extent,-extent,extent])
            ax[i].grid(False)
            cbar = fig.colorbar(im)
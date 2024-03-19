import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, TextBox

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


def viewAFM(img, vsize=1.0, interactive=False, interpolate=None):
    if isinstance(img , list):
        nimg = len(img)
    else:
        if len(img.shape) ==3:
            nimg = img.shape[0]
        else:
            nimg = 1
            img = [img]
    max_amp = max([img[i].max() for i in range(nimg)])/10.0
    if not interactive :
        fig, ax = plt.subplots(1, nimg, figsize = (5*nimg,4))
        if nimg == 1:
            extent = (vsize * img[0].shape[0]) /10.0
            im = ax.imshow(img[0].T/10.0, origin="lower", cmap="afmhot", vmin = 0.0, vmax = max_amp , aspect='auto',
                           extent=[0,extent,extent,0], interpolation=interpolate)
            ax.grid(False)
            ax.set_xlabel("nm")
            ax.set_ylabel("nm")
            cbar = fig.colorbar(im)
            cbar.set_label("nm")

        else:
            for i in range(nimg):
                extent = (vsize * img[i].shape[0])/10.0
                im = ax[i].imshow(img[i].T/10.0, origin="lower", cmap="afmhot", vmin = 0.0, vmax = max_amp, aspect='auto',
                                  extent=[0,extent,extent,0], interpolation=interpolate)
                ax[i].grid(False)
                ax[i].set_xlabel("nm")
                ax[i].set_ylabel("nm")
                cbar = fig.colorbar(im)
                cbar.set_label("nm")

    else:

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plt.subplots_adjust(bottom=0.2)
        extent = (vsize * img[0].shape[0])/10.0
        im1 = ax.imshow(img[0].T/10.0, origin="lower", cmap="afmhot", vmin = 0.0, vmax = max_amp, #aspect='auto',
                                  extent=[0,extent,0,extent], interpolation=interpolate)

        vinitax = plt.axes([0.6, 0.1, 0.05, 0.04])
        vinitBox = TextBox(vinitax, "", initial=str(0))
        axcolor = 'lightblue'

        def update(val):
            im1.set_data((img[int(vinitBox.text) + 0]).T /10.0)
            fig.canvas.draw_idle()

        doneax = plt.axes([0.7, 0.1, 0.1, 0.04])
        buttond = Button(doneax, 'Done', color=axcolor, hovercolor='0.975')
        rightax = plt.axes([0.5, 0.1, 0.1, 0.04])
        buttonr = Button(rightax, '->', color=axcolor, hovercolor='0.975')
        leftax = plt.axes([0.4, 0.1, 0.1, 0.04])
        buttonl = Button(leftax, '<-', color=axcolor, hovercolor='0.975')

        def right(event):
            vinit = int(vinitBox.text)
            if vinit != nimg -1:
                vinitBox.set_val(str(vinit + 1))
            update(0)

        def left(event):
            vinit = int(vinitBox.text)
            if vinit != 0:
                vinitBox.set_val(str(vinit - 1))
            update(0)

        def done(event):
            plt.close(fig)

        buttonr.on_clicked(right)
        buttonl.on_clicked(left)
        buttond.on_clicked(done)
        cbar = fig.colorbar(im1)

        plt.show(block=True)
    return fig, ax



def viewAFMfitting(fitset, pca=None,interpolate="bicubic"):
    stk = fitset.imglib.imgs
    stk_est = fitset.fitlib.imgs
    sim = fitset.simulator
    nimg = fitset.imglib.nimg
    size = sim.size
    vsize=sim.vsize
    max_amp = max([np.max(stk[0]), np.max(stk_est[0])])
    diff = (stk[0] - stk_est[0]).T
    max_diff = max(np.abs(diff.max()), np.abs(diff.min()))
    ncols = 5
    if pca is not None:
        ncols=6

    fig, ax = plt.subplots(1, ncols, figsize=(20, 5))
    plt.subplots_adjust(bottom=0.2)

    extent = vsize * size // 2
    im1 = ax[0].imshow(stk[0].T, origin="lower", cmap="afmhot", vmin = 0.0, vmax = max_amp, #aspect='auto',
                              extent=[-extent,extent,-extent,extent], interpolation=interpolate)
    im2 = ax[1].imshow(stk_est[0].T, origin="lower", cmap="afmhot", vmin = 0.0, vmax = max_amp, #aspect='auto',
                              extent=[-extent,extent,-extent,extent], interpolation=interpolate)
    im3 = ax[2].imshow(diff, origin="lower", cmap="jet", vmax=max_diff, vmin = -max_diff,#aspect='auto',
                              extent=[-extent,extent,-extent,extent], interpolation=interpolate)

    l1 = ax[3].plot(fitset.mse[0])
    l2 = ax[4].plot(fitset.rmsd[0])
    if pca is not None:
        ax[5].scatter(pca[:,0], pca[:,1])
        s = ax[5].scatter(pca[0,0], pca[0,1], c="r")
    cax = fig.add_axes([0.88, 0.2, 0.05, 0.6])
    cax.set_axis_off()
    cbar = fig.colorbar(im1, ax=cax)
    cax2 = fig.add_axes([0.92, 0.2, 0.05, 0.6])
    cax2.set_axis_off()
    cbar2 = fig.colorbar(im3, ax=cax2)

    vinitax = plt.axes([0.6, 0.1, 0.05, 0.04])
    vinitBox = TextBox(vinitax, "", initial=str(0))
    axcolor = 'lightblue'

    def update(val):
        max_amp = max([np.max(stk[int(vinitBox.text)]), np.max(stk_est[int(vinitBox.text)])])
        diff = (stk[int(vinitBox.text) + 0]-stk_est[int(vinitBox.text) + 0]).T
        max_diff = max(np.abs(diff.max()), np.abs(diff.min()))
        im1.set_data((stk[int(vinitBox.text) + 0]).T)
        im2.set_data((stk_est[int(vinitBox.text) + 0]).T)
        im3.set_data(diff)
        im1.set_clim(vmax=max_amp)
        im2.set_clim(vmax=max_amp)
        im3.set_clim(vmax=max_diff, vmin = -max_diff)
        cbar.update_normal(im1)
        cbar.update_ticks()
        cbar2.update_normal(im3)
        cbar2.update_ticks()
        mse = fitset.mse[int(vinitBox.text)]
        rmsd =fitset.rmsd[int(vinitBox.text)]
        l1[0].set_ydata(mse)
        l2[0].set_ydata(rmsd)
        ax[3].set_ylim(mse.min(), mse.max())
        ax[4].set_ylim(rmsd.min(), rmsd.max())
        if pca is not None:
            s.set_offsets(pca[int(vinitBox.text),:2])
        fig.canvas.draw_idle()

    doneax = plt.axes([0.7, 0.1, 0.1, 0.04])
    buttond = Button(doneax, 'Done', color=axcolor, hovercolor='0.975')
    rightax = plt.axes([0.5, 0.1, 0.1, 0.04])
    buttonr = Button(rightax, '->', color=axcolor, hovercolor='0.975')
    leftax = plt.axes([0.4, 0.1, 0.1, 0.04])
    buttonl = Button(leftax, '<-', color=axcolor, hovercolor='0.975')

    def right(event):
        vinit = int(vinitBox.text)
        if vinit != nimg -1:
            vinitBox.set_val(str(vinit + 1))
        update(0)

    def left(event):
        vinit = int(vinitBox.text)
        if vinit != 0:
            vinitBox.set_val(str(vinit - 1))
        update(0)

    def done(event):
        plt.close(fig)

    buttonr.on_clicked(right)
    buttonl.on_clicked(left)
    buttond.on_clicked(done)

    plt.show(block=True)
    return fig, ax




def compareFitting(fitset1,fitset2, pca=None,interpolate="bicubic"):
    stk1 = fitset1.fitlib.imgs
    stk2 = fitset2.fitlib.imgs
    stk_est = fitset1.imglib.imgs
    sim = fitset1.simulator
    nimg = fitset1.imglib.nimg
    size = sim.size
    vsize=sim.vsize
    max_amp = max([np.max(stk1[0]),np.max(stk2[0]), np.max(stk_est[0])])
    diff1 = -(stk1[0] - stk_est[0]).T
    diff2 = -(stk2[0] - stk_est[0]).T
    diff3 = (stk2[0] - stk1[0]).T
    max_diff = max([np.abs(diff1.max()), np.abs(diff1.min()), np.abs(diff2.max()), np.abs(diff2.min()),
                    np.abs(diff3.max()), np.abs(diff3.min())])
    ncols = 7
    if pca is not None:
        ncols=8

    fig, ax = plt.subplots(1, ncols, figsize=(25, 5))
    plt.subplots_adjust(bottom=0.2)

    extent = vsize * size // 2
    im1 = ax[0].imshow(stk1[0].T, origin="lower", cmap="afmhot", vmin = 0.0, vmax = max_amp, #aspect='auto',
                              extent=[-extent,extent,-extent,extent], interpolation=interpolate)
    im2 = ax[1].imshow(stk2[0].T, origin="lower", cmap="afmhot", vmin = 0.0, vmax = max_amp, #aspect='auto',
                              extent=[-extent,extent,-extent,extent], interpolation=interpolate)
    im3 = ax[2].imshow(stk_est[0].T, origin="lower", cmap="afmhot", vmin = 0.0, vmax = max_amp, #aspect='auto',
                              extent=[-extent,extent,-extent,extent], interpolation=interpolate)
    im4 = ax[3].imshow(diff1, origin="lower", cmap="jet", vmax=max_diff, vmin = -max_diff,#aspect='auto',
                              extent=[-extent,extent,-extent,extent], interpolation=interpolate)
    im5 = ax[4].imshow(diff2, origin="lower", cmap="jet", vmax=max_diff, vmin = -max_diff,#aspect='auto',
                              extent=[-extent,extent,-extent,extent], interpolation=interpolate)
    im6 = ax[5].imshow(diff3, origin="lower", cmap="jet", vmax=max_diff, vmin = -max_diff,#aspect='auto',
                              extent=[-extent,extent,-extent,extent], interpolation=interpolate)

    l2 = ax[6].plot(fitset2.mse[0], "o-")
    l1 = ax[6].plot(fitset1.mse[0], "o-")
    if pca is not None:
        ax[7].scatter(pca[:,0], pca[:,1])
        s = ax[7].scatter(pca[0,0], pca[0,1], c="r")
    cax = fig.add_axes([0.88, 0.2, 0.05, 0.6])
    cax.set_axis_off()
    cbar = fig.colorbar(im1, ax=cax)
    cax2 = fig.add_axes([0.92, 0.2, 0.05, 0.6])
    cax2.set_axis_off()
    cbar2 = fig.colorbar(im3, ax=cax2)

    vinitax = plt.axes([0.6, 0.1, 0.05, 0.04])
    vinitBox = TextBox(vinitax, "", initial=str(0))
    axcolor = 'lightblue'

    def update(val):
        idx= int(vinitBox.text)
        max_amp = max([np.max(stk1[idx]), np.max(stk2[idx]), np.max(stk_est[idx])])
        diff1 = -(stk1[idx] - stk_est[idx]).T
        diff2 = -(stk2[idx] - stk_est[idx]).T
        diff3 = (stk2[idx] - stk1[idx]).T
        max_diff = max([np.abs(diff1.max()), np.abs(diff1.min()), np.abs(diff2.max()), np.abs(diff2.min()),
                        np.abs(diff3.max()), np.abs(diff3.min())])

        im1.set_data((stk1[idx]).T)
        im2.set_data((stk2[idx]).T)
        im3.set_data((stk_est[idx]).T)
        im4.set_data(diff1)
        im5.set_data(diff2)
        im6.set_data(diff3)
        im1.set_clim(vmax=max_amp)
        im2.set_clim(vmax=max_amp)
        im3.set_clim(vmax=max_amp)
        im4.set_clim(vmax=max_diff, vmin = -max_diff)
        im5.set_clim(vmax=max_diff, vmin = -max_diff)
        im6.set_clim(vmax=max_diff, vmin = -max_diff)
        cbar.update_normal(im1)
        cbar.update_ticks()
        cbar2.update_normal(im4)
        cbar2.update_ticks()
        mse1 = fitset1.mse[idx]
        mse2 = fitset2.mse[idx]
        l1[0].set_ydata(mse1)
        l2[0].set_ydata(mse2)
        ax[6].set_ylim(min (mse2.min(), mse1.min()), max(mse1.max(), mse2.max()))
        if pca is not None:
            s.set_offsets(pca[idx,:2])
        fig.canvas.draw_idle()

    doneax = plt.axes([0.7, 0.1, 0.1, 0.04])
    buttond = Button(doneax, 'Done', color=axcolor, hovercolor='0.975')
    rightax = plt.axes([0.5, 0.1, 0.1, 0.04])
    buttonr = Button(rightax, '->', color=axcolor, hovercolor='0.975')
    leftax = plt.axes([0.4, 0.1, 0.1, 0.04])
    buttonl = Button(leftax, '<-', color=axcolor, hovercolor='0.975')

    def right(event):
        vinit = int(vinitBox.text)
        if vinit != nimg -1:
            vinitBox.set_val(str(vinit + 1))
        update(0)

    def left(event):
        vinit = int(vinitBox.text)
        if vinit != 0:
            vinitBox.set_val(str(vinit - 1))
        update(0)

    def done(event):
        plt.close(fig)

    buttonr.on_clicked(right)
    buttonl.on_clicked(left)
    buttond.on_clicked(done)

    plt.show(block=True)
    return fig, ax


class FittingPlotter:

    def __init__(self):
        self.fig = None
        self.ax = None
        self.imgf_p = None
        self.imge_p = None
        self.imgfe_p = None
        self.mse_p = None
        self.rmsd_p = None
        self.mse_a = None
        self.rmsd_a = None

    def start(self, imgf, imge, mse, rmsd, interactive = False):
        if interactive:
            plt.ion()
        size=imge.shape[0]
        vmax = max(imge.max(), imgf.max())
        fig, ax = plt.subplots(1, 7, figsize=(15, 4))
        self.imgf_p  = ax[0].imshow(imgf.T, origin="lower", cmap="afmhot", vmax = vmax)
        self.imge_p  = ax[1].imshow(imge.T, origin="lower", cmap="afmhot", vmax = vmax)
        self.imgfe_p = ax[2].imshow((imge -imgf).T, origin="lower", cmap="jet")
        self.mse_p  = ax[3].plot(np.arange(len(mse)), np.sqrt(mse), zorder=1)
        self.rmsd_p =ax[4].plot(np.arange(len(rmsd)), rmsd, zorder=1)
        self.imgfx  = ax[5].plot(imgf[size//2])
        self.imgex  = ax[5].plot(imge[size//2])
        self.imgfy  = ax[6].plot(imgf[:, size//2])
        self.imgey  = ax[6].plot(imge[:, size//2])
        ax[0].set_title("Fitted")
        ax[1].set_title("Target")
        ax[2].set_title("Diff")
        ax[3].set_title("RMSE")
        ax[4].set_title("RMSD ($\\AA$)")
        ax[5].set_title("Slice X")
        ax[6].set_title("Slice Y")

        self.fig=fig
        self.ax=ax
        fig.show()


    def update_imgs(self, imgf, imge):
        size=imge.shape[0]
        vmax = max(imge.max(), imgf.max())

        #imgs
        self.imgf_p.set_data(imgf.T)
        self.imge_p.set_data(imge.T)
        self.imgf_p.set_clim(vmax=vmax)
        self.imge_p.set_clim(vmax=vmax)
        self.imgfe_p.set_data((imge -imgf).T)
        self.imgfx[0].set_ydata(imgf[size//2])
        self.imgex[0].set_ydata(imge[size//2])
        self.imgfy[0].set_ydata(imgf[:, size//2])
        self.imgey[0].set_ydata(imge[:, size//2])

    def plot_attempt(self,length, mses, rmsds):
        n_lines = len(mses)
        size = len(mses[0])
        x = np.arange(length - size, length)
        self.mse_a = []
        self.rmsd_a = []
        for n in range(n_lines):
            l = self.ax[3].plot(x, np.sqrt(mses[n]), "o-", color="grey", zorder=0)
            self.mse_a.append(l)
            l = self.ax[4].plot(x, rmsds[n], "o-", color="grey", zorder=0)
            self.rmsd_a.append(l)

    def clear_attempt(self):
        if self.mse_a is not None:
            for i in range(len(self.mse_a)):
                self.mse_a[i][0].remove()
            self.mse_a = None
        if self.rmsd_a is not None:
            for i in range(len(self.rmsd_a)):
                self.rmsd_a[i][0].remove()
            self.rmsd_a = None

    def update(self, imgf, imge, mse, rmsd, mse_a=None, rmsd_a=None):

        #imgs
        self.update_imgs(imgf, imge)

        if mse_a is not None and rmsd_a is not None:
            self.clear_attempt()
            self.plot_attempt(len(mse), mse_a, rmsd_a)
        #mse
        x = np.arange(len(mse))
        minmse = np.min(np.sqrt(mse))
        maxmse = np.max(np.sqrt(mse))
        rangemse = (maxmse-minmse)
        minmse -= rangemse*0.1
        maxmse += rangemse*0.1
        self.mse_p[0].set_xdata(x)
        self.mse_p[0].set_ydata(np.sqrt(mse))
        self.ax[3].set_xlim(0,len(mse))
        self.ax[3].set_ylim(minmse,maxmse)
        #rmsd
        x = np.arange(len(rmsd))
        minrmsd = np.min(rmsd)
        maxrmsd = np.max(rmsd)
        rangermsd = (maxrmsd-minrmsd)
        minrmsd -= rangermsd*0.1
        maxrmsd += rangermsd*0.1
        self.rmsd_p[0].set_xdata(x)
        self.rmsd_p[0].set_ydata(rmsd)
        self.ax[4].set_xlim(0,len(rmsd))
        self.ax[4].set_ylim(minrmsd,maxrmsd)

        self.draw()


    def draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

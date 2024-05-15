#     AFMfit - Fitting package for Atomic Force Microscopy data
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, TextBox

def viewAFM(img, vsize=1.0, interactive=False, interpolate=None):
    """
    Display an AFM image or set of images
    :param img: Images or set of images
    :param vsize: pixel size
    :param interactive: if true, the images are shown one by one in an interactive plot
    :param interpolate: if defined, performs the interpolation (e.g. "spline36", "bicubic", "bilinear")
    :return: Matplotlib figure, axes
    """
    if isinstance(img , list):
        nimg = len(img)
    else:
        if len(img.shape) ==3:
            nimg = img.shape[0]
        else:
            nimg = 1
            img = [img]
    max_amp = max([img[i].max() for i in range(nimg)])/10.0
    if not interactive and nimg <10 :
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
        cbar.set_label("nm")

        plt.show(block=True)
    return fig, ax

def viewFit(fitter, pca=None,interpolate="bicubic", diff_range=None):
    """
    Display results of a fitting
    :param fitter: Fitter instance to display
    :param pca: PCA data if available
    :param interpolate:  if defined, performs the interpolation (e.g. "spline36", "bicubic", "bilinear")
    :param diff_range: Range in Ang of the difference plot (automated if not defined)
    :return: matplotlib Figure
    """
    stk1 = fitter.rigid_imgs
    stk_est = fitter.imgs
    sim     = fitter.simulator
    rigid_mse = fitter.rigid_mses
    if fitter.flexible_done:
        stk2 = fitter.flexible_imgs
        flexible_mse = fitter.flexible_mses
    else:
        stk2 = np.zeros(stk1.shape)
        flexible_mse = rigid_mse
    nimg = fitter.nimgs
    size = sim.size
    vsize=sim.vsize
    extent = vsize * size /10


    def get_imgs(idx):
        img1 = stk_est[idx].T /10
        img2 = stk1[idx].T /10
        img3 = stk2[idx].T /10
        max_amp = max([img1.max(), img2.max(),img3.max()])
        diff1 = (img2 - img3)
        diff2 = -(img2 -img1)
        diff3 = -(img3 - img1)
        max_diff = max([np.abs(diff1.max()), np.abs(diff1.min()), np.abs(diff2.max()), np.abs(diff2.min()),
                        np.abs(diff3.max()), np.abs(diff3.min())])
        return img1, img2,img3, diff1,diff2, diff3, max_amp, max_diff

    def get_ang(idx):
        def deg2rad(a):
            return (np.deg2rad(a) - np.pi) % (2 * np.pi) - np.pi
        ang1 = deg2rad(fitter.rigid_angles[idx, :, 0])
        ang2 = deg2rad(fitter.rigid_angles[idx, :, 1]) - np.pi / 2
        ang3 = deg2rad(fitter.rigid_angles[idx, :, 2])
        angc = fitter.rigid_mses[idx]
        if fitter.flexible_done:
            angf = [deg2rad(fitter.flexible_angles[idx, 0]), deg2rad(fitter.flexible_angles[idx, 1])- np.pi / 2, deg2rad(fitter.flexible_angles[idx, 2])]
        else:
            angf = [ang1[0], ang2[0], ang3[0]]
        angr = [ang1[0], ang2[0], ang3[0]]
        return ang1, ang2, ang3, angc, angf,angr

    img1, img2,img3, diff1,diff2, diff3, max_amp, max_diff =get_imgs(0)
    if diff_range is not None:
        max_diff = diff_range
    ang1, ang2, ang3, angc,angf,angr = get_ang(0)


    # Create fig
    fig = plt.figure(layout="constrained", figsize=(20, 6.8))
    gs = fig.add_gridspec(3, 2,height_ratios=[1,1,0.2])

    gsimg = gs[0].subgridspec(1, 3)
    gsdiff = gs[2].subgridspec(1, 3)
    gsstat = gs[1].subgridspec(1, 3)
    gsang = gs[3].subgridspec(1, 2)

    aximg = fig.add_subplot(gsimg[0, 0])
    aximr = fig.add_subplot(gsimg[0, 1])
    aximf = fig.add_subplot(gsimg[0, 2])

    axdif = fig.add_subplot(gsdiff[0, 0])
    axdfr = fig.add_subplot(gsdiff[0, 1])
    axdff = fig.add_subplot(gsdiff[0, 2])

    axmse = fig.add_subplot(gsstat[0, 0])
    axrms = fig.add_subplot(gsstat[0, 1])

    axan1 = fig.add_subplot(gsang[0, 0], projection="hammer")
    axan2 = fig.add_subplot(gsang[0, 1], projection="hammer")

    im1 = aximg.imshow(img1, origin="lower", cmap="afmhot", vmin = 0.0, vmax = max_amp,
                              extent=[0,extent,extent,0], interpolation=interpolate)
    im2 = aximr.imshow(img2, origin="lower", cmap="afmhot", vmin = 0.0, vmax = max_amp,
                              extent=[0,extent,extent,0], interpolation=interpolate)
    im3 = aximf.imshow(img3, origin="lower", cmap="afmhot", vmin = 0.0, vmax = max_amp,
                              extent=[0,extent,extent,0], interpolation=interpolate)
    im4 = axdif.imshow(diff1, origin="lower", cmap="jet", vmin = 0.0, vmax = max_amp,
                              extent=[0,extent,extent,0], interpolation=interpolate)
    im5 = axdfr.imshow(diff2, origin="lower", cmap="jet", vmin = -max_diff, vmax = max_diff,
                              extent=[0,extent,extent,0], interpolation=interpolate)
    im6 = axdff.imshow(diff3, origin="lower", cmap="jet", vmin = -max_diff, vmax = max_diff,
                              extent=[0,extent,extent,0], interpolation=interpolate)

    aximg.set_title("$I_{exp}$")
    aximr.set_title("$I_{rigid}$")
    aximf.set_title("$I_{flex}$")
    axdif.set_title("$|I_{rigid} - I_{flex}|$")
    axdfr.set_title("$|I_{rigid} - I_{exp}|$")
    axdff.set_title("$|I_{flex} - I_{exp}|$")
    aximg.set_ylabel("nm")
    axdif.set_ylabel("nm")
    axdif.set_xlabel("nm")
    axdfr.set_xlabel("nm")
    axdff.set_xlabel("nm")
    if fitter.flexible_done:
        lmsf = axmse.plot(np.sqrt(flexible_mse[0]), "o-")
        lmsr = axmse.plot(np.sqrt(rigid_mse[0,:flexible_mse.shape[1]]), "o-")
        lrms = axrms.plot(fitter.flexible_rmsds[0], "o-")
    else:
        lmsr = axmse.plot(np.sqrt(rigid_mse[0]), "o-")
    axmse.set_title("pixel-wise $l_2$")
    axmse.set_xlabel("iter")
    axrms.set_title("RMSD ($\AA$)")
    axrms.set_xlabel("iter")

    if pca is not None:
        axpca = fig.add_subplot(gsstat[0, 2])
        axpca.scatter(pca[:,0], pca[:,1])
        spca = axpca.scatter(pca[0,0], pca[0,1], c="r")
        axpca.set_title("PCA")
        axpca.set_xlabel("PC1")
        axpca.set_ylabel("PC2")

    cbar1 = fig.colorbar(im3, ax=aximf)
    cbar2 = fig.colorbar(im6, ax=axdff)
    cbar1.set_label("nm")
    cbar2.set_label("nm")

    axan1.xaxis.set_major_locator(plt.FixedLocator(np.pi / 3 * np.linspace(-2, 2, 5)))
    axan1.xaxis.set_minor_locator(plt.FixedLocator(np.pi / 6 * np.linspace(-5, 5, 11)))
    axan1.yaxis.set_major_locator(plt.FixedLocator(np.pi / 6 * np.linspace(-2, 2, 5)))
    axan1.yaxis.set_minor_locator(plt.FixedLocator(np.pi / 12 * np.linspace(-5, 5, 11)))
    axan1.grid(True, which='minor')
    san1 =axan1.scatter(ang3, ang2 , c=angc, cmap="viridis")
    san1r =axan1.scatter(angr[2], angr[1] , c="orange")
    san1f =axan1.scatter(angf[2], angf[1] , c="red")
    axan1.set_xlabel("Psi")
    axan1.set_ylabel("Tilt")


    axan2.xaxis.set_major_locator(plt.FixedLocator(np.pi / 3 * np.linspace(-2, 2, 5)))
    axan2.xaxis.set_minor_locator(plt.FixedLocator(np.pi / 6 * np.linspace(-5, 5, 11)))
    axan2.yaxis.set_major_locator(plt.FixedLocator(np.pi / 6 * np.linspace(-2, 2, 5)))
    axan2.yaxis.set_minor_locator(plt.FixedLocator(np.pi / 12 * np.linspace(-5, 5, 11)))
    axan2.grid(True, which='minor')
    san2 = axan2.scatter(ang1, ang2 , c=angc, cmap="viridis")
    san2r =axan2.scatter(angr[0], angr[1] , c="orange")
    san2f =axan2.scatter(angf[0], angf[1] , c="red")
    axan2.set_xlabel("Rot")
    axan2.set_ylabel("Tilt")

    # if cbar and color is not None:
    #     fig.colorbar(im2, ax=ax)


    vinitax = plt.axes([0.6, 0.02, 0.05, 0.04])
    vinitBox = TextBox(vinitax, "", initial=str(0))
    axcolor = 'lightblue'

    def update(val):
        idx= int(vinitBox.text)
        img1, img2, img3, diff1, diff2, diff3, max_amp, max_diff = get_imgs(idx)
        ang1, ang2, ang3, angc,angf, angr = get_ang(idx)
        if diff_range is not None:
            max_diff = diff_range
        im1.set_data(img1)
        im2.set_data(img2)
        im3.set_data(img3)
        im4.set_data(diff1)
        im5.set_data(diff2)
        im6.set_data(diff3)
        im1.set_clim(vmax=max_amp)
        im2.set_clim(vmax=max_amp)
        im3.set_clim(vmax=max_amp)
        im4.set_clim(vmax=max_diff, vmin = -max_diff)
        im5.set_clim(vmax=max_diff, vmin = -max_diff)
        im6.set_clim(vmax=max_diff, vmin = -max_diff)
        cbar1.update_normal(im1)
        cbar1.update_ticks()
        cbar2.update_normal(im4)
        cbar2.update_ticks()
        if fitter.flexible_done:
            mser =np.sqrt(rigid_mse[idx, :flexible_mse.shape[1]])
            msef =np.sqrt(flexible_mse[idx])
            rmsd = fitter.flexible_rmsds[idx]
            lmsr[0].set_ydata(mser)
            lmsf[0].set_ydata(msef)
            lrms[0].set_ydata(rmsd)
            axmse.set_ylim(min(mser.min(), msef.min()), max(mser.max(), msef.max()))
            axrms.set_ylim(rmsd.min(), rmsd.max())
        else:
            mser =np.sqrt(rigid_mse[idx])
            lmsr[0].set_ydata(mser)
            axmse.set_ylim(mser.min(),mser.max())
        if pca is not None:
            spca.set_offsets(pca[idx,:2])
        san1.set_offsets( np.array([ang3, ang2]).T)
        san1r.set_offsets( np.array([angr[2], angr[1]]))
        san1f.set_offsets( np.array([angf[2], angf[1]]))
        san2r.set_offsets( np.array([angr[0], angr[1]]))
        san2f.set_offsets( np.array([angf[0], angf[1]]))
        san2.set_offsets( np.array([ang1, ang2]).T)
        fig.canvas.draw_idle()

    doneax = plt.axes([0.7, 0.02, 0.1, 0.04])
    buttond = Button(doneax, 'Done', color=axcolor, hovercolor='0.975')
    rightax = plt.axes([0.5, 0.02, 0.1, 0.04])
    buttonr = Button(rightax, '->', color=axcolor, hovercolor='0.975')
    leftax = plt.axes([0.4, 0.02, 0.1, 0.04])
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
    return fig


class FittingPlotter:

    def __init__(self):
        """
        DEPRECATED
        """
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

def show_angular_distr(angles, color=None, cmap = "jet", proj = "hammer", cbar = False, vmax=None):
    """
    Display a flatten projection of angular distribution of a set of Euler angles in the sphere
    :param angles: set of angles N * 3
    :param color: array of N with values to assign to each set fo angle
    :param cmap: colormap (e.g. "viridis")
    :param proj: projection method (e.g. "hammer", "lambert")
    :param cbar: If True, show the colorbar
    :param vmax: Maximum value for the colorbar
    """
    a1 = (np.deg2rad(angles[:,0])-np.pi)%(2*np.pi) -np.pi
    a2 = (np.deg2rad(angles[:,1])-np.pi)%(2*np.pi) -np.pi
    a3 = (np.deg2rad(angles[:,2])-np.pi)%(2*np.pi) -np.pi

    fig = plt.figure(figsize=(10, 4))
    ax = plt.subplot(121, projection=proj)
    ax.xaxis.set_major_locator(plt.FixedLocator(np.pi / 3  * np.linspace(-2, 2, 5 )))
    ax.xaxis.set_minor_locator(plt.FixedLocator(np.pi / 6  * np.linspace(-5, 5, 11)))
    ax.yaxis.set_major_locator(plt.FixedLocator(np.pi / 6  * np.linspace(-2, 2, 5 )))
    ax.yaxis.set_minor_locator(plt.FixedLocator(np.pi / 12 * np.linspace(-5, 5, 11)))
    ax.grid(True, which='minor')
    im1 = ax.scatter(a3, a2-np.pi/2, c=color, cmap=cmap, vmax=vmax)
    ax.set_xlabel("Psi")
    ax.set_ylabel("Tilt")
    ax = plt.subplot(122, projection=proj)
    ax.xaxis.set_major_locator(plt.FixedLocator(np.pi / 3  * np.linspace(-2, 2, 5 )))
    ax.xaxis.set_minor_locator(plt.FixedLocator(np.pi / 6  * np.linspace(-5, 5, 11)))
    ax.yaxis.set_major_locator(plt.FixedLocator(np.pi / 6  * np.linspace(-2, 2, 5 )))
    ax.yaxis.set_minor_locator(plt.FixedLocator(np.pi / 12 * np.linspace(-5, 5, 11)))
    ax.grid(True, which='minor')
    im2 = ax.scatter(a1, a2-np.pi/2, c=color, cmap=cmap, vmax=vmax)
    ax.set_xlabel("Rot")
    ax.set_ylabel("Tilt")
    if cbar and color is not None:
        fig.colorbar(im2, ax=ax)

    plt.show()

# def show_fe(data, size, interpolate=None, cmap="jet"):
#     xmin = np.min(data[:, 0])
#     xmax = np.max(data[:, 0])
#     ymin = np.min(data[:, 1])
#     ymax = np.max(data[:, 1])
#     xm = (xmax - xmin) * 0.1
#     ym = (ymax - ymin) * 0.1
#     xmin -= xm
#     xmax += xm
#     ymin -= ym
#     ymax += ym
#     x = np.linspace(xmin, xmax, size)
#     y = np.linspace(ymin, ymax, size)
#     count = np.zeros((size, size))
#     for i in range(data.shape[0]):
#         count[np.argmin(np.abs(x.T - data[i, 0])),
#         np.argmin(np.abs(y.T - data[i, 1]))] += 1
#     img = -np.log(count / count.max())
#     img[img == np.inf] = img[img != np.inf].max()
# 
#     fig, ax = plt.subplots(1,1)
#     if interpolate is not None:
#         im = ax.imshow(img.T[::-1, :],
#                        cmap=cmap, interpolation=interpolate,
#                        extent=[xmin, xmax, ymin, ymax])
#     else:
#         xx, yy = np.mgrid[xmin:xmax:size * 1j, ymin:ymax:size * 1j]
#         im = ax.contourf(xx, yy, img, cmap=cmap, levels=12)
#     #
#     cbar = fig.colorbar(im)
#     cbar.set_label("$\Delta G / k_{B}T$")
#     fig.show()


from skimage.filters import gaussian
from scipy.ndimage import binary_erosion, binary_dilation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from tiffile import imwrite, imread
from afmfit.viewer import viewAFM, show_angular_distr
from numba import njit
import mrcfile



def mask_interactive(stk):
    vinit =0
    mask = mask_stk(stk, itere=2, iterd=2, threshold=30.0, sigma=1.0)

    fig, ax = plt.subplots(1,3, figsize=(10,5))
    plt.subplots_adjust(bottom=0.5)
    im1 = ax[0].imshow((stk[vinit +0] * mask[vinit + 0]).T, origin="lower", cmap="afmhot")
    im2 = ax[1].imshow((stk[vinit +1] * mask[vinit + 1]).T, origin="lower", cmap="afmhot")
    im3 = ax[2].imshow((stk[vinit +2] * mask[vinit + 2]).T, origin="lower", cmap="afmhot")
    # ax.margins(x=0)

    axcolor = 'lightblue'
    as_thres = plt.axes([0.2, 0.35, 0.65, 0.03], facecolor=axcolor)
    ax_sigma = plt.axes([0.2, 0.30, 0.65, 0.03], facecolor=axcolor)
    ax_itere = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_iterd = plt.axes([0.2, 0.25, 0.65, 0.03], facecolor=axcolor)

    thres = Slider(as_thres, 'Threshold (A)', 0.0, 100.0, valinit=30.0, valstep=1.0)
    sigma = Slider(ax_sigma, 'Gaussian filter', 0.1, 10.0, valinit=1.0, valstep=0.1)
    itere = Slider(ax_itere, 'Erosion', 0, 10, valinit=2, valstep=1)
    iterd = Slider(ax_iterd, 'Dilatation', 0, 10, valinit=2, valstep=1)
    vinitax = plt.axes([0.6, 0.1, 0.05, 0.04])
    vinitBox= TextBox(vinitax, "", initial=str(vinit))

    def update(val):
        mask = mask_stk(stk, itere=itere.val, iterd=iterd.val, threshold=thres.val, sigma=sigma.val)
        im1.set_data((stk[int(vinitBox.text) + 0] * mask[int(vinitBox.text)+0]).T)
        im2.set_data((stk[int(vinitBox.text) + 1] * mask[int(vinitBox.text)+1]).T)
        im3.set_data((stk[int(vinitBox.text) + 2] * mask[int(vinitBox.text)+2]).T)
        fig.canvas.draw_idle()

    thres.on_changed(update)
    sigma.on_changed(update)
    itere.on_changed(update)
    iterd.on_changed(update)

    resetax = plt.axes([0.2, 0.1, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    doneax = plt.axes([0.7, 0.1, 0.1, 0.04])
    buttond = Button(doneax, 'Done', color=axcolor, hovercolor='0.975')
    rightax = plt.axes([0.5, 0.1, 0.1, 0.04])
    buttonr = Button(rightax, '->', color=axcolor, hovercolor='0.975')
    leftax = plt.axes([0.4, 0.1, 0.1, 0.04])
    buttonl = Button(leftax, '<-', color=axcolor, hovercolor='0.975')

    def reset(event):
        thres.reset()
        sigma.reset()
        itere.reset()
        iterd.reset()

    def right(event):
        vinit = int(vinitBox.text)
        if vinit != stk.shape[0] - 4:
            vinitBox.set_val(str(vinit +1))
        update(0)
    def left(event):
        vinit = int(vinitBox.text)
        if vinit != 0:
            vinitBox.set_val(str(vinit-1))
        update(0)

    def done(event):
        plt.close(fig)

    button.on_clicked(reset)
    buttonr.on_clicked(right)
    buttonl.on_clicked(left)
    buttond.on_clicked(done)


    plt.show(block=True)
    return mask_stk(stk, itere=itere.val, iterd=iterd.val, threshold=thres.val, sigma=sigma.val)

# class AFMImage:
#     def __init__(self, img, vsize):
#         self.img = img
#         self.vsize = vsize
#
#
#     @classmethod
#     def from_mrc(cls, filename):
#         pass
#     @classmethod
#     def from_tif(cls, filename):
#         pass

def mask_stk(stk, itere = 1, iterd = 2,threshold = 40.0,sigma = 1.0) :
    mask_stk = np.zeros(stk.shape)
    for i in range(stk.shape[0]):
        g = gaussian(stk[i], sigma=sigma)
        binary_mask = g > threshold
        binary_mask = binary_erosion(binary_mask, iterations=itere)
        binary_mask = binary_dilation(binary_mask, iterations=iterd)
        mask_stk[i] = binary_mask
    return mask_stk

class ImageSet:

    def __init__(self, imgs, vsize=1.0, angles=None, shifts=None):
        self.imgs = np.array(imgs, dtype=np.float32)
        self.nimg = self.imgs.shape[0]
        self.size = self.imgs.shape[1]
        self.vsize = vsize
        if angles is None:
            self.angles = np.zeros((self.nimg, 3))
        else:
            self.angles=angles
        if shifts is None:
            self.shifts = np.zeros((self.nimg, 3))
        else:
            self.shifts=shifts
    @classmethod
    def read_tif(cls, file, vsize=1.0, unit='nm'):
        arr = imread(file).astype(np.float32)
        img = cls.arr2img(arr, unit)

        print("Read %i images of size %i x %i "%img.shape)
        return cls(img, vsize)
    @classmethod
    def read_mrc(cls, file, vsize=1.0, unit='nm'):
        with mrcfile.open(file) as mrc:
            arr = np.array(mrc.data, dtype=np.float32)
        img = cls.arr2img(arr, unit)

        print("Read %i images of size %i x %i "%img.shape)
        return cls(img, vsize)

    def write_tif(self, file, unit = "nm"):
        arr = self.img2arr(self.imgs, unit)
        imwrite(file, arr)

    def write_mrc(self, file, unit = "nm"):
        arr = self.img2arr(self.imgs, unit)
        with mrcfile.new(file, overwrite=True) as mrc:
            mrc.set_data(arr)
            mrc.voxel_size = self.vsize
            mrc.update_header_from_data()
            mrc.update_header_stats()

    def get_imgs(self):
        return self.imgs

    @classmethod
    def arr2img(self, arr, unit):
        if len(arr.shape) ==2:
            img = np.array([arr], dtype=np.float32)
        else:
            img = np.array(arr, dtype=np.float32)
        if unit == "m" or unit == "meter":
            img*= 1.0e10
        if unit == "mm" or unit == "millimeter":
            img*= 1.0e7
        if unit == "um" or unit == "micrometer":
            img*= 1.0e4
        if unit == "nm" or unit == "nanometer":
            img*= 10.0
        elif unit == "ang"or unit == "angstrom":
            img*= 1.0
        img = img.transpose(0,2,1)[:,:,::-1]
        return img
    @classmethod
    def img2arr(self, imgs, unit):
        arr = np.array(imgs, dtype=np.float32)
        if unit == "m" or unit == "meter":
            arr/= 1.0e10
        if unit == "mm" or unit == "millimeter":
            arr/= 1.0e7
        if unit == "um" or unit == "micrometer":
            arr/= 1.0e4
        if unit == "nm" or unit == "nanometer":
            arr/= 10.0
        elif unit == "ang"or unit == "angstrom":
            arr/= 1.0
        arr = arr[:,:,::-1].transpose(0,2,1)
        return arr

    def show(self, **kwargs):
        viewAFM(self.imgs, interactive=True,**kwargs)

    def show_angular_distr(self, **kwargs):
        show_angular_distr(self.angles,**kwargs)

class ImageLibrary:

    def __init__(self, imgRawArray, nimgs, size, vsize, view_group=None, angles=None, z_shifts=None):
        self.imgRawArray = imgRawArray
        self.nimgs = nimgs
        self.size = size
        self.vsize = vsize
        self.angles = angles
        self.z_shifts = z_shifts
        self.norm2 = self.calculate_norm(self.get_imgs())
        if view_group is None:
            self.ngroup = None
            self.nview = None
        else:
            self.nview, self.ngroup = view_group.shape
        self.view_group = view_group

    def get_imgs(self):
        return np.frombuffer(self.imgRawArray, dtype=np.float32,
                               count=len(self.imgRawArray)).reshape(self.nimgs, self.size,self.size)

    def get_img(self, index):
        return self.get_imgs()[index]
    # return frombuffer(self.imgRawArray, offset=index*self.size*self.size*4, dtype=np.float32,
                     #count=self.size*self.size).reshape(self.size,self.size)

    def calculate_norm(self, imgs):
        return calculate_norm_njit(imgs)
    def show(self, **kwargs):
        viewAFM(self.get_imgs(), interactive=True, **kwargs)

    def show_angular_distr(self, **kwargs):
        show_angular_distr(self.angles, **kwargs)

@njit
def calculate_norm_njit(imgs):
    nimg = imgs.shape[0]
    norm2 = np.zeros(nimg)
    for i in range(nimg):
        norm2[i] = np.sum(np.square(imgs[i]))
    return norm2


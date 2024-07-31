import multiprocessing
import os
import sys

import numpy as np
import mrcfile
from afmfit.pdbio import PDB,  dcd2numpyArr, numpyArr2dcd
from afmfit.nma import NormalModesRTB
from afmfit.simulator import AFMSimulator, sigma_estimate
from afmfit.viewer import viewAFM, viewFit, show_angular_distr
from afmfit.image import mask_stk, mask_interactive, Particles
from afmfit.fitting import Fitter
from afmfit.utils import  DimRed, align_coords,get_sphere_full, get_angular_distance, \
    get_init_angles_flat
import matplotlib.pyplot as plt
import matplotlib
from afmfit.image import ImageSet
import trackpy as tp
import tkinter as tk
import tkinter.ttk as ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
from tkinter.filedialog import askopenfilename, asksaveasfilename, askdirectory
from afmfit.utils import get_flattest_angles, check_chimerax_in_path
import copy
from tkinter import messagebox
from tkinter import scrolledtext
from functools import partial
import multiprocessing
import traceback
import pickle
import tqdm
import time
from afmfit.utils import run_chimerax
import threading
import afmfit

small_font = ("calibri", 12)
default_font = ("calibri", 15)
large_font = ("calibri", 18)

bg_color = '#d9d9d9'
button_color ='#000000'
completed_color = "#458B00"
failed_color = "#CD3333"

SCALE_WIDTH = 400


class AFMfitImsetFrame:
    def __init__(self, imset, frame, update_callback=None, **kwargs):
        self.imset = copy.deepcopy(imset)
        self.fig = None
        self.im = None
        self.cbar = None
        self.currentVar = None

        self.update_callback= update_callback

        self.setup_imageFrame(frame, **kwargs)
    def update_imageFrame(self, event=None):
        self.im.set_data((self.imset.get_img(self.get_current())).T / 10.0)
        self.im.autoscale()
        self.cbar.update_normal(self.im)
        self.cbar.update_ticks()
        self.fig.canvas.draw_idle()

        if self.update_callback is not None:
            self.update_callback()

    def get_current(self):
        if self.currentVar is None:
            return 0
        else:
            return self.currentVar.get()-1

    def setup_imageFrame(self, frame, interpolate="spline36", name=""):
        extentx = (self.imset.vsize * self.imset.sizex) / 10.0
        extenty = (self.imset.vsize * self.imset.sizey) / 10.0
        self.fig = plt.Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        self.im = self.ax.imshow(self.imset.get_img(0).T / 10.0, origin="lower", cmap="afmhot",
                       extent=[0, extentx, extenty, 0], interpolation=interpolate)
        self.ax.grid(False)
        self.ax.set_xlabel("nm")
        self.ax.set_ylabel("nm")
        self.cbar = self.fig.colorbar(self.im)
        self.cbar.set_label("nm")
        self.ax.set_title(name)

        canvas = FigureCanvasTkAgg(self.fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)

        def update_image(event=None):
            self.update_imageFrame()

        def right():
            vinit = self.currentVar.get()
            if vinit != self.imset.nimg - 1:
                self.currentVar.set(vinit+1)
            update_image()

        def left():
            vinit = self.currentVar.get()
            if vinit != 0:
                self.currentVar.set(vinit-1)
            update_image()

        controlsFrame = tk.Frame(master=frame, bg=bg_color)
        left_button = tk.Button(master=controlsFrame, command=left, height=1, width=10, text="<")
        right_button = tk.Button(master=controlsFrame, command=right, height=1, width=10, text=">")
        self.currentVar = tk.IntVar()
        self.currentVar.set(1)
        currentScale = tk.Scale(controlsFrame, from_=1, to=self.imset.nimg, variable=self.currentVar,
                             orient=tk.HORIZONTAL, label="", resolution=1
                             , command=self.update_imageFrame, length=SCALE_WIDTH)

        if self.imset.nimg >1:
            controlsFrame.grid(row=1, column=0)
            left_button.grid(row=0, column=0)
            right_button.grid(row=0, column=1)
            currentScale.grid(row=0, column=2)

        toolbarFrame = tk.Frame(master=frame, background=bg_color)
        toolbarFrame.grid(row=2, column=0, sticky="s")
        toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)


class AFMfitViewer:
    def __init__(self, imset):
        self.input_imset = imset

    def setup_window(self, toplevel=None):
        if toplevel is not None:
            self.window = tk.Toplevel(toplevel)
        else:
            self.window = tk.Tk()
        self.window.title('AFMFit viewer')
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=2)
        self.window.configure(background=bg_color)

    def setup_frames(self,**kwargs):
        mplFrame = tk.Frame(master=self.window)
        mplFrame.grid(row=0, column=0)
        self.mainImageFrame = AFMfitImsetFrame(self.input_imset, mplFrame, **kwargs)

    def view(self,toplevel=None, **kwargs):
        self.setup_window(toplevel)
        self.setup_frames(**kwargs)
        if toplevel is None:
            self.window.mainloop()
        else:
            self.window.transient(toplevel)
            self.window.grab_set()
            toplevel.wait_window(self.window)

    def quit(self):
        self.window.destroy()

class AFMfitPreprocessing(AFMfitViewer):
    def __init__(self, imset):
        super().__init__(imset)

    def setup_frames(self,**kwargs):
        mplFrame = tk.Frame(master=self.window)
        mplFrame.grid(row=0, column=0)
        self.mainImageFrame = AFMfitImsetFrame(self.input_imset, mplFrame, update_callback= None ,**kwargs)
        self.setup_processingFrame()

    def setup_processingFrame(self):
        toolsFrame = tk.Frame(master=self.window)
        toolsFrame.grid(row=0, column=1, sticky="N")
        tk.Label(toolsFrame, text="Image Processing tools", font=default_font).grid(row=0, column=0, sticky="W")



        calibFrame = tk.Frame(master=toolsFrame)
        calibFrame.grid(row=1, column=0, sticky="W")
        self.minVar = tk.StringVar()
        self.maxVar = tk.StringVar()

        tk.Label(calibFrame, text="Min", width=3).grid(row=0, column=0)
        self.minEntry = tk.Entry(calibFrame, textvariable=self.minVar, width=6)
        self.minEntry.grid(row=0, column=1)
        tk.Label(calibFrame, text="Max", width=3).grid(row=0, column=2)
        self.maxEntry = tk.Entry(calibFrame, textvariable=self.maxVar, width=6)
        self.maxEntry.grid(row=0, column=3)
        tk.Button(master=calibFrame, text="Apply",
                  command=self.calibrateUpdate).grid(row=0, column=4)

        tk.Button(master=toolsFrame, text="Reset",
                  command=self.resetUpdate).grid(row=3, column=0, sticky="W")

        tk.Button(master=toolsFrame, command=self.quit, height=1, width=10,
                  text="Done", fg=completed_color).grid(row=10, column=0, sticky="SE")
        tk.Button(master=toolsFrame, text="Normalize",
                  command=self.normalizeUpdate).grid(row=2, column=0, sticky="W")

        self.resetUpdate()

    def normalizeUpdate(self):
        self.mainImageFrame.imset.normalize_mode()
        self.update_processingFrame()
    def calibrateUpdate(self):
        self.mainImageFrame.imset.set_min(10.0*float(self.minVar.get()))
        self.mainImageFrame.imset.set_max(10.0*float(self.maxVar.get()))
        self.update_processingFrame()

    def resetUpdate(self):
        minval = "%.2f"% (self.mainImageFrame.imset.get_imgs().min()/10.0)
        maxval = "%.2f"% (self.mainImageFrame.imset.get_imgs().max()/10.0)
        self.minVar.set(minval)
        self.maxVar.set(maxval)
        self.minEntry.delete(0, tk.END)
        self.minEntry.insert(0, minval)
        self.maxEntry.delete(0, tk.END)
        self.maxEntry.insert(0, maxval)
        self.mainImageFrame.imset = copy.deepcopy(self.input_imset)
        self.update_processingFrame()
    def update_processingFrame(self):
        self.mainImageFrame.update_imageFrame()

class AFMfitPicker(AFMfitViewer):
    def __init__(self, imset):
        super().__init__(imset)
        self.picking_sc = None
        self.box_sc = None
        self.particles = None

        self.sepaScal = None
        self.maxsizeScaleVar = None
        self.minmassScale = None
        self.diamScale = None
    def get_particles(self):
        return self.particles

    def get_centroids(self, ):
        frame = self.mainImageFrame.imset.get_img(self.mainImageFrame.get_current()).T[::-1]
        data = tp.locate(frame, **self.get_trackpy_args())
        vsize= self.mainImageFrame.imset.vsize/10.0
        return np.array(data["x"])*vsize, (np.array(data["y"])*vsize)

    def update_pickingFrame(self, event=None):
        centroids = self.get_centroids()
        offset_update = np.array([centroids[0], centroids[1]]).T
        self.picking_sc.set_offsets(offset_update)
        self.box_sc.set_offsets(offset_update)
        self.mainImageFrame.fig.canvas.draw_idle()

    def extract_particles(self):
        frames = self.mainImageFrame.imset.get_imgs()
        centroids = tp.batch(frames, **self.get_trackpy_args(), processes=1)
        centers = []
        for i in range(self.mainImageFrame.imset.nimg):
            centers_frame = []
            idx = np.where(i == centroids["frame"])[0]
            for j in idx:
                centers_frame.append([centroids["y"][j], centroids["x"][j]])
            centers.append(centers_frame)
        self.particles = Particles(self.mainImageFrame.imset, centers, boxsize=self.get_boxsize())

        self.quit()

    def get_trackpy_args(self):
        maxsize = self.maxsizeScaleVar.get() * 10.0 / self.mainImageFrame.imset.vsize
        return {
            "diameter": self.get_diameter(),
            "invert": False,
            "separation": self.sepaScale.get() * 10.0 / self.mainImageFrame.imset.vsize,
            "minmass": self.minmassScale.get(),
            "maxsize": maxsize if maxsize != 0 else None
        }

    def get_diameter(self):
        diam = int(self.diamScale.get() * 10.0 / self.mainImageFrame.imset.vsize)
        if diam % 2 == 0:
            diam += 1
        return diam

    def get_boxsize(self):
        box = int(self.boxScale.get() * 10.0 / self.mainImageFrame.imset.vsize)
        if box % 2 != 0:
            box += 1
        return box

    def update_box(self, event):
        self.box_sc.set_sizes(np.ones(len(self.box_sc.get_sizes())) * (self.get_boxsize() * self.mainImageFrame.imset.vsize) ** 2 / 20)
        self.update_pickingFrame()

    def update_pad(self, event):
        margin = self.padScale.get()
        imgs = self.input_imset.get_imgs()
        if margin != 0:
            nimg, sizex, sizey = imgs.shape
            new_imgs = np.zeros((nimg, sizex + 2 * margin, sizey + 2 * margin))
            new_imgs[:, margin:-margin, margin:-margin] = imgs
        else:
            new_imgs = imgs
        self.mainImageFrame.imset = ImageSet(new_imgs, vsize=self.mainImageFrame.imset.vsize)

        extentx = (self.mainImageFrame.imset.vsize * self.mainImageFrame.imset.sizex + 2 * margin) / 10.0
        extenty = (self.mainImageFrame.imset.vsize * self.mainImageFrame.imset.sizey + 2 * margin) / 10.0
        self.mainImageFrame.im.set_extent([0.0, extentx, extenty, 0.0])
        self.mainImageFrame.update_imageFrame()

    def setup_pickingFrame(self):

        def reset():
            self.diamScale.set(10)
            self.sepaScale.set(20)
            self.minmassScale.set(1)
            self.boxScale.set(10)
            self.maxsizeScaleVar.set(0)
            self.padScale.set(0)

        imset = self.mainImageFrame.imset
        toolsFrame = tk.Frame(master=self.window)
        toolsFrame.grid(row=0, column=1, sticky="N")
        trackpyFrame = tk.Frame(master=toolsFrame, background=bg_color)
        trackpyFrame.grid(row=0, column=0, sticky="W")
        trackpylabel = tk.Label(trackpyFrame, text="Picking tools", font=default_font)
        trackpylabel.grid(row=0, column=0, sticky="W")
        self.diamScale = tk.Scale(trackpyFrame, from_=1, to=imset.vsize * min(imset.sizex, imset.sizey) // 2 / 10.0 * 2,
                             orient=tk.HORIZONTAL, label="Diameter (nm)", resolution=1
                             , command=self.update_pickingFrame, length=SCALE_WIDTH)
        self.diamScale.grid(row=2, column=0, sticky="W")
        self.sepaScale = tk.Scale(trackpyFrame, from_=1, to=imset.vsize * min(imset.sizex, imset.sizey) // 2 / 10.0,
                             orient=tk.HORIZONTAL, label="Separation (nm)", resolution=1
                             , command=self.update_pickingFrame, length=SCALE_WIDTH)
        self.sepaScale.grid(row=3, column=0, sticky="W")
        self.minmassScale = tk.Scale(trackpyFrame, from_=0, to=10000,
                                orient=tk.HORIZONTAL, label="Min mass", resolution=1
                                , command=self.update_pickingFrame, length=SCALE_WIDTH)
        self.minmassScale.grid(row=4, column=0, sticky="W")

        self.maxsizeScaleVar = tk.DoubleVar()
        maxsizeScale = tk.Scale(trackpyFrame, from_=0, to=imset.vsize * min(imset.sizex, imset.sizey) // 2 / 10.0,
                                orient=tk.HORIZONTAL, label="Max size (nm)", resolution=1.0,
                                command=self.update_pickingFrame, length=SCALE_WIDTH, variable=self.maxsizeScaleVar)
        maxsizeScale.grid(row=5, column=0, sticky="W")
        self.boxScale = tk.Scale(trackpyFrame, from_=1, to=imset.vsize * min(imset.sizex, imset.sizey) // 2 / 10.0 * 2,
                            orient=tk.HORIZONTAL, label="Box size (nm)", resolution=1
                            , command=self.update_box, length=SCALE_WIDTH)
        self.boxScale.grid(row=6, column=0, sticky="W")
        self.padScale = tk.Scale(trackpyFrame, from_=0, to=50,
                            orient=tk.HORIZONTAL, label="Margin (px)", resolution=1
                            , command=self.update_pad, length=SCALE_WIDTH)
        self.padScale.grid(row=7, column=0, sticky="W")

        extractButton = tk.Button(master=toolsFrame, command=self.extract_particles, height=1, width=10, text="Done", fg=completed_color)
        extractButton.grid(row=1, column=0, sticky="W")

        reset()
        centroids = self.get_centroids()
        self.picking_sc = self.mainImageFrame.ax.scatter(centroids[0], centroids[1], s=5 * imset.vsize, facecolors='none',
                                edgecolors="fuchsia", marker="o")
        self.box_sc = self.mainImageFrame.ax.scatter(centroids[0], centroids[1], s=self.get_boxsize() * imset.vsize, facecolors='none',
                            edgecolors="lime", marker="s", alpha=0.5)
        self.update_pickingFrame()

    def setup_frames(self,**kwargs):
        mplFrame = tk.Frame(master=self.window)
        mplFrame.grid(row=0, column=0)
        self.mainImageFrame = AFMfitImsetFrame(self.input_imset, mplFrame, update_callback= self.update_pickingFrame ,**kwargs)
        self.setup_pickingFrame()


class AFMfitSimulatorViewer(AFMfitViewer):
    def __init__(self, pdb, init_sim, refimage=None):
        super().__init__(ImageSet(np.zeros((1,init_sim.size,init_sim.size)),init_sim.vsize))
        self.pdb=pdb
        self.refimage = refimage
        self.sim = None
        self.init_sim=init_sim


    def setup_simulatorFrame(self, toolsFrame):
        self.sim_sigma = tk.DoubleVar()
        self.sim_quality = tk.DoubleVar()
        self.sim_sigma.set(self.init_sim.sigma)
        self.sim_quality.set(8.0)

        self.update_simulator()
        self.update_image()


        simulatorFrame = tk.Frame(master=toolsFrame)
        simulatorFrame.grid(row=0, column=0, sticky="W")

        tk.Label(simulatorFrame, text="Simulator tools",font=default_font).grid(row=0, column=0, sticky="W")

        tk.Scale(simulatorFrame, from_=1.0,
                                  to=15.0, variable=self.sim_sigma,
                                  orient=tk.HORIZONTAL, label="Smoothness", resolution=0.1
                                  , command=self.update_simulatorFrame, length=SCALE_WIDTH).grid(row=1, column=0, sticky="W")

        tk.Button(master=simulatorFrame, command=self.autoSigma, height=1, width=10, text="Auto").grid(row=2, column=0, sticky="W")


        tk.Scale(simulatorFrame, from_=1.0,
                                  to=13.0, variable=self.sim_quality,
                                  orient=tk.HORIZONTAL, label="Quality", resolution=0.1
                                  , command=self.update_simulatorFrame, length=SCALE_WIDTH).grid(row=3, column=0, sticky="W")


        pdbFrame = tk.Frame(master=toolsFrame)
        pdbFrame.grid(row=1, column=0, sticky="W")

        tk.Label(pdbFrame, text="PDB tools", font=default_font).grid(row=0, column=0, sticky="W")

        tk.Button(master=pdbFrame, command=self.centerPDB, height=1, width=10, text="Center")\
            .grid(row=1, column=0, sticky="W")
        tk.Button(master=pdbFrame, command=self.orientPDB, height=1, width=10, text="Orient")\
            .grid(row=1, column=1, sticky="W")

        tk.Button(master=toolsFrame, command=self.quit, height=1, width=10, text="Done", fg=completed_color)\
            .grid(row=2, column=0, sticky="W")

        self.mainImageFrame.update_imageFrame()

    def get_sim(self):
        return self.sim
    def centerPDB(self, event=None):
        self.pdb.center()
        self.update_image()
        self.mainImageFrame.update_imageFrame()
    def orientPDB(self, event=None):
        angle = get_flattest_angles(self.pdb, percent=0.0, angular_dist=20)
        self.pdb.rotate(angle[0])

        self.update_image()
        self.mainImageFrame.update_imageFrame()
    def update_simulator(self):
        self.sim = AFMSimulator( size=self.init_sim.size,
                                 vsize=self.init_sim.vsize,
                                sigma=float(self.sim_sigma.get()),
                                 quality_ratio=float(self.sim_quality.get()) )
    def update_image(self):
        im = self.sim.pdb2afm(self.pdb)
        self.mainImageFrame.imset = ImageSet(np.array([im]), vsize=self.init_sim.vsize)
    def update_simulatorFrame(self, event=None):
        self.update_simulator()
        self.update_image()
        self.mainImageFrame.update_imageFrame()
    def autoSigma(self):
        params = ParamWindow(
            "Parameters",
            self.window,
            {
                "n_cpu": ["Num of CPUs", "Integer", 4, "Number of available CPU cores"],
                "resolution": ["Resolution", "Float", 0.5, "Resolution of estimation of sigma (minimum spacing)"],
                "angular_dist": ["Angular dist. between reprojections (°)", "Integer", 10, ""],
                "angle_precent": ["% angle", "Float", 20.0, ""],
                "lower_dmax": ["% of Dmax lower", "Float", 0.1, ""],
                "upper_dmax": ["% of Dmax upper", "Float", 1.0, ""],
                "max_nangle": ["max num angle", "Integer",100, ""],
                "max_sigma": ["max sigma", "Float", 15.0, ""],
                "min_sigma": ["min sigma", "Float", 2.0, ""],
            }
        ).get_params()

        sigma = sigma_estimate(imgs = self.refimage, pdb=self.pdb, quality_ratio=float(self.sim_quality.get()),**params)
        self.sim_sigma.set(sigma)
        self.update_simulatorFrame()
    def setup_frames(self, **kwargs):
        mplFrame = tk.Frame(master=self.window)
        mplFrame.grid(row=0, column=0)
        self.mainImageFrame = AFMfitImsetFrame(self.input_imset, mplFrame,
                                                update_callback=self.update_image, **kwargs)
        toolid = 1
        if self.refimage is not None:
            refFrame = tk.Frame(master=self.window)
            refFrame.grid(row=0, column=1)
            self.refImageFrame = AFMfitImsetFrame(self.refimage, refFrame, **kwargs)
            toolid+= 1
        toolsFrame = tk.Frame(master=self.window)
        toolsFrame.grid(row=0, column=toolid, sticky="N")
        self.setup_simulatorFrame(toolsFrame)


class AFMFitMenu:
    def __init__(self):
        # Buttons
        self.actionButtons = []
        self.viewButtons = []
        # Tk
        self.console_redirector = None
        self.window = None

        self.data={
            "state"         : 0,
            "failed"        : False
        }

        #Chimerax
        self.path_chimerax = None

    def get_state(self):
        return self.data["state"]
    def get_status(self):
        return self.data["failed"]
    def set_state(self, state):
        self.data["state"] = state
    def set_status(self, status):
        self.data["failed"] = status

    def dump(self):
        file = asksaveasfilename(filetypes=[ ('Pickle files', '*.pkl')])
        if file != "":
            with open(file, "wb") as f:
                pickle.dump(self.data, f)
            messagebox.showinfo("Info","Saved session : %s"%file)
        else:
            messagebox.showwarning("Warning","Could not save session")

    def load(self):
        file = askopenfilename(filetypes=[ ('Pickle files', '*.pkl')])
        if file != "":
            with open(file, "rb") as f:
                self.data = pickle.load(f)
            self.update_actionButton()
            messagebox.showinfo("Info","Loaded session : %s"%file)
        else:
            messagebox.showwarning("Warning","Could not load session")


    def update_actionButton(self):
        nactions = len(self.actionButtons)

        # NORMAL/DISABLED
        for i in range(nactions):
            if i<=self.get_state():
                self.actionButtons[i].config(state=tk.NORMAL)
                self.viewButtons[i].config(state=tk.NORMAL)
            else:
                self.actionButtons[i].config(state=tk.DISABLED)
                self.viewButtons[i].config(state=tk.DISABLED)

        #colors
        for i in range(nactions):
            if i<self.get_state():
                self.actionButtons[i].config(fg=completed_color)
            else:
                if i==self.get_state() and self.get_status():
                    self.actionButtons[i].config(fg=failed_color)
                else:
                    self.actionButtons[i].config(fg=button_color)


    def setup_window(self):
        self.window = tk.Tk()
        self.window.title('AFMFit')
        # self.window.geometry("500x700")
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        self.window.configure(background=bg_color)

    def view(self):
        self.setup_window()

        ioFrame = tk.Frame(master=self.window)
        ioFrame.grid(row=0, column=0, sticky="W")
        self.setup_io(ioFrame)

        chimeraxFrame = tk.Frame(master=self.window)
        chimeraxFrame.grid(row=0, column=1)
        self.setup_chimerax(chimeraxFrame)

        menuFrame = tk.Frame(master=self.window)
        menuFrame.grid(row=1, column=0, columnspan=2)
        self.setup_menuFrame(menuFrame)

        consoleFrame = tk.Frame(master=self.window)
        consoleFrame.grid(row=2, column=0, columnspan=2, sticky="N")
        self.setup_console(consoleFrame)

        self.welcome()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()


    def setup_console(self, frame):
        console_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, bg=button_color, fg=bg_color, width=55)
        console_text.grid(row=0, column=0)
        self.console_redirector = ConsoleRedirector(console_text)

    def setup_io(self, frame):
        loadButton = tk.Button(text="Load session", command=self.load, master=frame)
        saveButton = tk.Button(text="Save session", command=self.dump, master=frame)

        loadButton.grid(row=0, column=0)
        saveButton.grid(row=0, column=1)

    def setup_chimerax(self, frame):
        try :
            self.path_chimerax = check_chimerax_in_path()
            print("Chimerax detected")
        except RuntimeError:
            print("ChimeraX not found")
        if self.path_chimerax is not None:
            color = completed_color
        else:
            color = failed_color
        self.chimeraxLabel = tk.Label(text="ChimeraX",master=frame, fg=color)
        self.chimeraxLabel.grid(row=0, column=0, sticky="E")

    def welcome(self):
        print("""
                  ______ __  __    __ _ _   
            /\   |  ____|  \/  |  / _(_) |  
           /  \  | |__  | \  / | | |_ _| |_ 
          / /\ \ |  __| | |\/| | |  _| | __|
         / ____ \| |    | |  | | | | | | |_ 
        /_/    \_\_|    |_|  |_| |_| |_|\__|

        ---------------------------------------                                    
               """)

        print("Running AFMfit version %s "%afmfit.__version__)


    def get_steps(self):
        return {
            "importAFM"      : ["Import AFM data", self.importAFMdata, self.viewAFMdata],
            "preprocess"     : ["Preprocessing", self.preprocess, self.viewAFMdata],
            "picking"        : ["Pick particles",  self.picking, self.viewParticles],
            "importPDB"      : ["Import PDB", self.importPDB, self.viewPDB],
            "simulator"      : ["Setup simulator", self.setupSimulator, self.viewSimulator],
            "calculateNMA"   : ["Calculate NMA", self.calculateNMA, self.viewNMA],
            "rigidFit"       : ["Rigid Fitting", self.rigidFit, self.viewrigidFit],
            "flexibleFit"    : ["Flexible Fitting", self.flexibleFit, self.viewrigidFit],
            "exportMovies"   : ["Export Fitted Images", self.exportMovies, self.viewMovies],
            "exportPDBs"     : ["Export Fitted PDBs", self.exportPDBs, self.viewPDBs],
            "analyzeEnsemble": ["Analyze ensemble", self.analyzeEnsemble, self.viewEnsemble],
            "extractMotions" : ["Extract motions",self.extractMotions, self.viewMotions],
        }

    def setup_menuFrame(self, frame):
        steps = self.get_steps()
        for i,k in enumerate(steps):
            name = steps[k][0]
            stepfunc = partial(self.run_step, i, steps[k][1])
            viewfunc = steps[k][2]
            step_button = tk.Button(text=name, command=stepfunc,  master=frame,  height=1, width=30, font=default_font)
            view_button = tk.Button( command=viewfunc, master=frame,  height=1, width=5, font=default_font,text="View")
            step_button.grid(row=i, column=0, sticky="W")
            view_button.grid(row=i, column=1, sticky="W")
            self.actionButtons.append(step_button)
            self.viewButtons.append(view_button)
        self.update_actionButton()

    def run_step(self, state, step_function):
        self.set_state(state)
        try :
            status = step_function()
            self.window.config(cursor="")
        except Exception as e:
            self.window.config(cursor="")
            messagebox.showerror("Error",str(e))
            traceback.print_exc()
            self.set_status(True)
        else:
            if status == 0 or status is None:
                self.set_status(False)
                self.set_state(1+self.get_state())
            else:
                self.set_status(True)
                messagebox.showerror("Error",str(status))

        self.update_actionButton()
    def importAFMdata(self):
        file = askopenfilename(filetypes=[ ('ASD files', '*.asd'), ('TIFF files', '*.tif *.tiff'), ('Text files', '*.txt')])

        if os.path.isfile(file):
            name, ext = os.path.splitext(file)
            if ext == ".tiff" or ext == ".tif" or ext == ".txt":
                params = ParamWindow(
                    "Enter pixel size",
                    self.window,
                    {
                        "vsize": ["Pixel Size (nm)", "Float", 1.0, "Specify the pixel size of the AFm data"],
                        "unit": ["Z-scale unit (nm)", "Float", 1.0, "Specify the unit of the zscale. 1.0 = Nanometer, 0.1 = Angstrom, 1e9 = meter "],
                    }
                ).get_params()
                if ext == ".tiff" or ext == ".tif" :
                    self.data["imset"] = ImageSet.read_tif(file, vsize=params["vsize"]*10.0,unit=params["unit"]*10.0)
                else:
                    self.data["imset"] = ImageSet.read_txt(file, vsize=params["vsize"]*10.0,unit=params["unit"]*10.0)
            elif ext == ".asd":
                self.data["imset"] = ImageSet.read_asd(file)
            else:
                return "Could not open AFM file"

    def importPDB(self):
        file = askopenfilename(filetypes=[('PDB file', '*.pdb')])
        if os.path.isfile(file):
            name, ext = os.path.splitext(file)
            if ext == ".pdb":
                pdb = PDB(file)
                self.data["pdb"] = pdb
                print("Loaded PDB file : %s "%file)
                print("the file contains %i atoms"%pdb.n_atoms)
            else:
                return "File format not supported"
        if "pdb" not in self.data:
            return "No PDB file selected"

    def preprocess(self):
        preprocessor = AFMfitPreprocessing(self.data["imset"])
        preprocessor.view(toplevel=self.window)
        self.data["imset"] = preprocessor.mainImageFrame.imset

    def picking(self):
        picker = AFMfitPicker(self.data["imset"])
        picker.view(toplevel=self.window)

        particles = picker.get_particles()
        if particles is None:
            return "No particle selected"
        self.data["particles"] = particles
    def setupSimulator(self):
        simview = AFMfitSimulatorViewer(self.data["pdb"], init_sim=AFMSimulator(
            size=self.data["particles"].sizex,
            vsize=self.data["particles"].vsize,
            beta=1.0,
            sigma=5.0,
            cutoff=40.0
        ), refimage=self.data["particles"])
        simview.view(toplevel=self.window)
        simulator = simview.get_sim()

        if simulator is None:
            return "Could not get the simulator"

        self.data["pdb"] = simview.pdb
        self.data["simulator"] = simulator
    def calculateNMA(self):
        params = ParamWindow(
            "NMA params",
            self.window,
            {
                "nmodes": ["Number of Modes", "Integer", 10, "Number of normal modes to calculate"],
                "cutoff": ["Cutoff (Ang)", "Float", 8.0, "Cutoff distance in Angstrom for the elastic network model"],
            }).get_params()

        self.window.config(cursor="watch")
        self.window.update()

        self.data["nma"] = NormalModesRTB.calculate_NMA(pdb=self.data["pdb"], nmodes=params["nmodes"],cutoff =params["cutoff"] )
        if self.data["nma"] is None:
            return "NMA are not defined"


    def rigidFit(self):
        fitter= Fitter(pdb=self.data["pdb"], imgs=self.data["particles"].imgs, simulator=self.data["simulator"])
        params = ParamWindow(
            "Rigid Fit params",
            self.window,
            {
                "ncpu": ["Number of CPUs", "Integer", multiprocessing.cpu_count()//2, "Number of CPU cores to use"],
                "angular_dist": ["Angular distance (°)", "Float", 10.0, "Minimum angular distance between projected views"],
                "flatonly": ["Flat orientation only", "Bool", True, "Use only projection views that keeps the model flat relative to the Z-axis"],
                "flatper": ["% of flatness", "Float", 10.0, "Keep only the N% flattest angles", "flatonly"],
                "zshift_range": ["Z-shift range (Ang)", "Float", 20.0,
                             "Range of search in the z-shift direction"],
                "zshift_resolution": ["Z-shift resolution (points)", "Integer", 10,
                             "Resolution of search in the z-shift direction, more points gives better resolution"],
                "true_zshift": ["Z-shift projection", "Bool", False,
                             "Performs a true projection for each Z-shift points, more time consuming"],
            }
        ).get_params()
        if params["flatonly"]:
            angle = get_flattest_angles(self.data["pdb"], angular_dist=int(params["angular_dist"]), percent=params["flatper"])
        else:
            angle = None
        self.window.config(cursor="watch")
        self.window.update()
        fitter.fit_rigid(n_cpu=params["ncpu"],
                         angular_dist=int(params["angular_dist"]),
                         verbose=True,
                         zshift_range=np.linspace(-params["zshift_range"]//2,
                                                  params["zshift_range"]//2,
                                                  params["zshift_resolution"]),
                         init_zshift=None,
                         init_angles = angle,
                         select_view_group = True,
                         true_zshift=params["true_zshift"],
                         init_library=None)

        self.data["fitter"] = fitter

    def flexibleFit(self):
        fitter= self.data["fitter"]
        params = ParamWindow(
            "Flexible Fit params",
            self.window,
            {
                "ncpu": ["Number of CPUs", "Integer", multiprocessing.cpu_count()//2, "Number of CPU cores to use"],
                "n_iter": ["Num. of iterations of the algorithm", "Integer", 10,
                                 "The defautl value is typically sufficient to converge. Fewer iteration reduce computation time."],
                "lambda": ["Lambda", "Float", 10.0,
                                 "Parameter that controls the balance between data fitting and prior structure."
                                 " Larger values leads to smaller data fitting. Smaller values may overfit."],
                "n_best_views": ["Num. of projection views", "Integer", 10,
                                 "Run a flexible fitting for the number of best projection views selected."],
                "dist_views": ["Angle dist. between projection views", "Float",15,
                               "Restrict the fit to projection views that are at least distant from the specified value in degrees."],

            }
        ).get_params()

        self.window.config(cursor="watch")
        self.window.update()

        fitter.fit_flexible(n_cpu=params["ncpu"],
                            nma=self.data["nma"],
                            n_best_views=params['n_best_views'],
                            dist_views=params['dist_views'],
                            n_iter=params['n_iter'],
                            lambda_r=params['lambda']**2,  #
                            lambda_f=params['lambda']**2,
                            verbose=True)

        self.data["fitter"] = fitter
    def exportMovies(self):
        outparticles = copy.copy(self.data["particles"])
        outparticles.imgs = self.data["fitter"].flexible_imgs
        reconstructed = outparticles.place_to_centers()

        file = asksaveasfilename(filetypes=[ ('TIFF file', '*.tif')])
        if file != "":
            reconstructed.write_tif(file)
            messagebox.showinfo("Info","Saved image : %s"%file)
        else:
            messagebox.showwarning("Warning","Could not save image")

        self.data["reconstructed"] = reconstructed


    def exportPDBs(self):
        params = ParamWindow(
            "Parameters",
            self.window,
            {
                "type": ["Output type", "Enum", "DCD", "Chose the output file type", ["DCD", "PDB"]],

            }
        ).get_params()

        file = asksaveasfilename()
        if file == "":
            return "Must provide a prefix"
        self.window.config(cursor="watch")
        self.window.update()

        fitter= self.data["fitter"]
        pdb = fitter.pdb.copy()
        if params["type"] == "DCD":
            pdb.write_pdb("%s.pdb"%(file))
            numpyArr2dcd(fitter.flexible_coords, "%s.dcd"%(file))
        else:
            for i in range(fitter.nimg):
                pdb.coords = fitter.flexible_coords[i]
                pdb.write_pdb("%s%s.pdb"%(file, str(i+1).zfill(5)))

        messagebox.showinfo("Info", "Saved PDBs : %s" % file)

    def analyzeEnsemble(self):
        params = ParamWindow(
            "Parameters",
            self.window,
            {
                "n_components": ["Number of PCA components", "Integer", 10, ""],
                "method": ["Method", "Enum", "PCA", "principal component analysis (PCA) or "
                                                    "uniform manifold approximation and projection (UMAP)",
                           ["PCA", "UMAP"]],

            }
        ).get_params()

        self.window.config(cursor="watch")
        self.window.update()
        self.data["dimred"] = DimRed.from_fitter(self.data["fitter"], n_components=params["n_components"], method=params["method"])

    def extractMotions(self):
        params = ParamWindow(
            "Parameters",
            self.window,
            {
                "axis": ["Axis", "Integer", 1, "Axis along which the trajectory is drawn"],
                "n_points": ["Number of points", "Integer", 10, "Num of points in the trajectory"],
                "traj": ["Spacing", "Enum", "Linear", "spacing of points in the trajectory", ["Linear", "Percentiles"]],
                "avg": ["Reconstruction method", "Enum", "Inverse", "Inverse (PCA only) :  calculate inverse projection from PCA space to 3D coordinates"
                                                                    "Average : the closest conformation from each points of the trajectory are averaged", ["Inverse", "Average"]],
                "align": ["Align PDBs ?", "Bool", "True", "Rigid alignement of the fitted models to the reference"],

            }
        ).get_params()
        print(params)

        prefix = asksaveasfilename()
        if prefix == "":
            return "Must provide a prefix"

        self.window.config(cursor="watch")
        self.window.update()

        cluster = None
        if params["traj"] =="Linear":
            traj = self.data["dimred"].traj_linear(ax=params["axis"]-1, n_points=params["n_points"], method="max")
        else:
            traj = self.data["dimred"].traj_linear(ax=params["axis"]-1, n_points=params["n_points"])

        if params["avg"] =="Average":
            cluster =  self.data["dimred"].traj2cluster(traj)
            traj = None

        self.data["dimred"].output_traj(traj=traj, cluster=cluster, align=params["align"], align_ref=None, prefix=prefix)

        self.data["dimred"].show(cval=cluster, points=traj)

        self.data["motions_outfile"] = prefix
    def viewAFMdata(self):
        AFMfitViewer(self.data["imset"]).view(toplevel=self.window)

    def viewPDB(self):
        self.ask_chimerax()
        print("hello")
        print(self.path_chimerax)
        self.data["pdb"].viewChimera(self.path_chimerax)
    def viewParticles(self):
        AFMfitViewer(self.data["particles"]).view(toplevel=self.window)

    def viewNMA(self):
        self.ask_chimerax()

        params= ParamWindow(
            "NMA viewer parameters",
            self.window,
            {
                "amp": ["Amplitude", "Float", 1000.0, "Amplitude of deformation along each normal mode"],
                "npoints": ["Number of points", "Integer", 10, "Number of points per trajectories"],
            }).get_params()
        self.data["nma"].viewChimera(amp=params["amp"], npoints=params["npoints"],
                                     path_chimerax=self.path_chimerax)
    def viewSimulator(self):
        simview = AFMfitSimulatorViewer(self.data["pdb"], init_sim=self.data["simulator"])
        simview.view(toplevel=self.window)

    def viewrigidFit(self):
        self.data["fitter"].show()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.console_redirector.restore_stdout()
            self.window.destroy()

    def ask_chimerax(self):
        if self.path_chimerax is None:
            try :
                self.path_chimerax= check_chimerax_in_path()
            except RuntimeError:
                messagebox.showwarning("Warning", "ChimeraX not found in PATH")
                path_chimerax=askopenfilename()
                if os.system("%s --version"%path_chimerax) ==0:
                    self.chimeraxLabel.config(fg=completed_color)
                    self.path_chimerax = path_chimerax
                else:
                    messagebox.showerror("Error", "Could not run Chimerax at %s"%path_chimerax)

    def viewMovies(self):
        AFMfitViewer(self.data["reconstructed"]).view(toplevel=self.window)
    def viewEnsemble(self):
        params = ParamWindow(
            "Parameters",
            self.window,
            {
                "axis": ["Components", "String", "1 2", "Select exactly two components to display in the x and y axis"],
                "alpha": ["Alpha", "Float", 0.8, "Transparency of the points between 0 and 1"],
            }
        ).get_params()

        ax1 = int(params['axis'].split(" ")[0])-1
        ax2 = int(params['axis'].split(" ")[1])-1

        self.data["dimred"].show(ax=[ax1, ax2], alpha=params["alpha"])

    def viewPDBs(self):
        pass
    def viewMotions(self):
        self.ask_chimerax()

        prefix = self.data["motions_outfile"]

        with open(prefix + "traj.cxc", "w") as f:
            f.write("open %straj*.pdb \n" % (prefix))
            f.write("morph all \n")
        run_chimerax(prefix + "traj.cxc", self.path_chimerax)
class ParamWindow:
    def __init__(self,name, toplevel, params):
        self.params = params
        self.name = name
        self.toplevel = toplevel

    def info(self, text):
        messagebox.showinfo("Info", text)


    def start(self):
        self.window = tk.Toplevel(self.toplevel)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.title(self.name)
        self.window.transient(self.toplevel)
        self.window.grab_set()

        self.setup_params()

        self.toplevel.wait_window(self.window)

    def get_name(self,k):
        return self.params[k][0]
    def get_type(self,k):
        return self.params[k][1]
    def get_default(self,k):
        return self.params[k][2]
    def get_info(self,k):
        return self.params[k][3]
    def get_choices(self,k):
        if self.get_type(k) == "Enum":
            return self.params[k][4]
        elif self.get_type(k) == "Bool":
            return ["True", "False"]
        else:
            raise RuntimeError("Not Enum")

    def get_params(self):
        self.start()
        out_params = {}
        for i,k in enumerate(self.params):
            val = self.tkparams[i].get()
            ptype = self.get_type(k)
            if ptype == "Float":
                out_params[k] = float(val)
            elif ptype== "Integer":
                out_params[k] = int(val)
            elif ptype == "String":
                out_params[k] = val
            elif ptype == "Enum":
                out_params[k] = val
            elif ptype == "Bool":
                out_params[k] = bool(val)
        return out_params

    def setup_params(self):
        self.tkparams = []
        for i,k in enumerate(self.params):
            var = tk.StringVar()
            var.set(str(self.get_default(k)))
            self.tkparams.append(var)
            tk.Label(self.window, text=self.get_name(k)).grid(row=i, column=0)
            if self.get_type(k) == "Enum" or self.get_type(k) == "Bool" :
                for j,c in enumerate(self.get_choices(k)):
                    tk.Radiobutton(self.window, variable=var, text=c, value=c).grid(row=i, column=1+j)
                tk.Button(self.window, command=partial(self.info, self.get_info(k)), text='Info').grid(row=i, column=j+2)
            else:
                tk.Entry(self.window, textvariable=var).grid(row=i, column=1)
                tk.Button(self.window, command=partial(self.info, self.get_info(k)) , text='Info').grid(row=i, column=2)
        tk.Button(self.window, command=self.window.destroy , text='Done').grid(row=len(self.params), column=0)


    def on_closing(self):
        self.window.destroy()

class ConsoleRedirector:
    def __init__(self, console_text):
        self.console_text = console_text
        self.stdout_orig = sys.stdout
        self.stderr_orig = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, text):
        if text != "\n":
            if "\r" in text:
                index = text.index("\r")
                self.console_text.delete("end-1c linestart",tk.END)
                # self.console_text.insert(tk.INSERT, text[:index])
                self.console_text.insert(tk.INSERT, text[index+1:])
                self.console_text.insert(tk.INSERT, "\n")
                self.console_text.insert(tk.INSERT, "\n")
            else:
                self.console_text.insert(tk.END, text)

            self.console_text.see(tk.END)
            self.console_text.update_idletasks()

    def flush(self):
        pass

    def restore_stdout(self):
        sys.stdout=self.stdout_orig
        sys.stderr=self.stderr_orig

if __name__ == "__main__":
    menu = AFMFitMenu()
    menu.view()




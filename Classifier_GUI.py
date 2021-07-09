#Written by Shayan Yazdani

import tkinter
from tkinter import *
from tkinter import filedialog, messagebox
import pandas as pd
from scipy.io import wavfile
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from ClassifyCalls import Classify

class Window:
        def __init__(self, master):
            self.master = master
            master.title("Rat Vocalization Classifier")
            self.boxfile=""
            self.wavefile=""
            self.currentnumber=0
            self.Spectrogarms=""
            self.Categories=""
            self.img=''
            self.str=StringVar()
            self.cat_entry = Entry(root,textvariable=self.str, width=19)
            self.cat_entry.place(x=236, y=50)
            self.bbutton= Button(root, text="Load wave file", command=self.loadwave, height = 1, width = 14,bg="steelblue").place(x=580, y=60)
            self.bbutton= Button(root, text="Load boxes file", command=self.loadboxes,  height = 1, width = 14,bg="steelblue")
            self.bbutton.place(x=580, y=20)
            self.bbutton= Button(root, text="Classify", command=self.applyclassifier,  height = 1, width = 14, bg="steelblue")
            self.bbutton.place(x=580, y=100)
            self.bbutton= Button(root, text="Save Categories", command=self.saveCat, height = 1, width = 14, bg="steelblue")
            self.bbutton.place(x=580, y=140)
            self.bbutton= Button(root, text="Next Call", command=self.next_call, height = 1, width = 12, bg="steelblue")
            self.bbutton.place(x=295, y=475)
            self.bbutton= Button(root, text="Previous Call", command=self.pre_call, height = 1, width = 12, bg="steelblue")
            self.bbutton.place(x=195, y=475)
            self.bbutton= Button(root, text="Change Category", command=self.get_cat, height = 1, width = 16, bg="steelblue")
            self.bbutton.place(x=235, y=20)


        def loadboxes(self):

            Tk().withdraw()
            self.filename = filedialog.askopenfilename()
            if self.filename:
                self.boxfile = pd.read_csv(self.filename)
                self.Spectrogarms=""
                self.Categories=""
                self.currentnumber=0



        def loadwave(self):

            Tk().withdraw()
            self.filename2 = filedialog.askopenfilename()
            if self.filename2:
                self.fs, self.wavefile = wavfile.read(self.filename2)



        def applyclassifier(self):

            if self.fs:
                self.Categories, self.Spectrogarms = Classify(self.fs, self.wavefile, self.boxfile)
                self.plot_sig()
            else:
                messagebox.showerror("Error", "Please load inputs")


        def next_call(self):

            if(len(self.Categories) > 0):
                if self.currentnumber<(len(self.Categories)-1):
                   self.currentnumber=self.currentnumber+1
                   self.plot_sig()


        def pre_call(self):

            if(len(self.Categories) > 0):
                if self.currentnumber>0:
                   self.currentnumber=self.currentnumber-1
                   self.plot_sig()


        def plot_sig(self):

            figure = Figure(figsize=(4, 4), dpi=100)
            plot1= figure.add_subplot(1, 1, 1)
            plot1.set_title("Category: "+self.Categories[self.currentnumber],color="white")
            self.img=self.Spectrogarms[self.currentnumber]
            self.rsz()
            plot1.imshow(self.img)
            plot1.axis('off')
            figure.set_facecolor((.18, .18, .18))
            canvas = FigureCanvasTkAgg(figure, root)
            canvas.get_tk_widget().place(x=90, y=75)


        def saveCat(self):

            filepath1 = filedialog.asksaveasfile(defaultextension=".csv")
            if filepath1:
                pd.DataFrame({'ID': list(range(1, len(self.Categories)+1)) ,'Categories':self.Categories},columns=['ID','Categories']).to_csv(filepath1.name,index=False)


        def get_cat(self):

            self.Categories[self.currentnumber]=self.cat_entry.get()
            self.plot_sig()


        def rsz(self):

            if np.shape(self.img)[0]<500:
                a=(500-np.shape(self.img)[0])//2
                b=(500-np.shape(self.img)[0])//2
                npad = ((a,b), (0, 0))
                self.img = np.pad(self.img, pad_width=npad, mode='constant', constant_values=0)

            if np.shape(self.img)[1]<800:
                a=(800-np.shape(self.img)[1])//2
                b=(800-np.shape(self.img)[1])//2
                npad = ((0,0),(a,b))
                self.img = np.pad(self.img, pad_width=npad, mode='constant', constant_values=0)


if __name__ == "__main__":
    root = Tk()
    root.configure(background='gray18')
    root.geometry("700x600")
    window=Window(root)
    #window.plot_sig()
    root.mainloop()

import os
import datetime
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Menu, Toplevel, messagebox
from tkinter import ttk

# Try importing SSIM
try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

class EdgeComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Matching App")
        self.root.geometry("1400x850")

        # Professional ttk style
        style = ttk.Style()
        style.theme_use('clam')

        # Variables
        self.master_img = None
        self.target_img = None
        self.master_gray = None
        self.target_gray = None
        self.roi_coords = None
        self.rect = None
        self.target_rect = None
        self.last_match_img = None
        self.algorithm = tk.StringVar(value="Combined")
        self.orb_features = tk.IntVar(value=500)
        self.orb_weight = tk.DoubleVar(value=0.5)
        self.show_keypoints = tk.BooleanVar(value=False)
        self.brightness = tk.IntVar(value=0)
        self.contrast = tk.DoubleVar(value=1.0)

        # Menu bar
        menubar = Menu(root)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Master...", accelerator="Ctrl+O", command=self.upload_master)
        file_menu.add_command(label="Open Target...", accelerator="Ctrl+T", command=self.upload_target)
        file_menu.add_separator()
        file_menu.add_command(label="Save Report...", accelerator="Ctrl+R", command=self.save_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Settings")

        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        root.config(menu=menubar)
        root.bind_all('<Control-o>', lambda e: self.upload_master())
        root.bind_all('<Control-t>', lambda e: self.upload_target())
        root.bind_all('<Control-r>', lambda e: self.save_report())

        # Paned window
        paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Scrollable sidebar
        sidebar_container = ttk.Frame(paned)
        paned.add(sidebar_container, weight=0)
        self.sidebar_canvas = tk.Canvas(sidebar_container, width=150)
        scrollbar = ttk.Scrollbar(sidebar_container, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar_frame = ttk.Frame(self.sidebar_canvas)
        self.sidebar_frame.bind(
            "<Configure>",
            lambda e: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
        )
        self.sidebar_canvas.create_window((0, 0), window=self.sidebar_frame, anchor='nw')
        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Main frame
        self.main_frame = ttk.Frame(paned, padding=10)
        paned.add(self.main_frame, weight=1)

        # Image canvases
        self.master_canvas = tk.Canvas(self.main_frame, width=600, height=500, bg='black')
        self.target_canvas = tk.Canvas(self.main_frame, width=600, height=500, bg='black')
        self.master_canvas.grid(row=0, column=0, padx=5, pady=5)
        self.target_canvas.grid(row=0, column=1, padx=5, pady=5)
        self.master_canvas.create_text(10, 10, anchor='nw', text='Master', fill='white', font=('Arial', 16, 'bold'))
        self.target_canvas.create_text(10, 10, anchor='nw', text='Target', fill='white', font=('Arial', 16, 'bold'))

        # Similarity label
        self.result_label = ttk.Label(self.main_frame, text="Similarity Score: N/A", font=('Arial', 14))
        self.result_label.grid(row=1, column=0, columnspan=2, pady=10)

        # Status bar
        self.status = ttk.Label(root, text="application", relief=tk.SUNKEN, anchor='w')
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # Progress bar
        self.progress = ttk.Progressbar(self.sidebar_frame, mode='indeterminate')
        self.progress.pack(fill='x', pady=1)

        # Sidebar controls
        controls = [
            (ttk.Button, {'text':"Upload Master Image", 'command':self.upload_master}),
            (ttk.Button, {'text':"Upload Target Image", 'command':self.upload_target}),
            (ttk.Button, {'text':"Draw ROI on Master", 'command':self.draw_roi_on_master}),
            (ttk.Button, {'text':"Show ROI on Target", 'command':self.draw_roi_on_target}),
            (ttk.Button, {'text':"Clear ROI", 'command':self.clear_roi}),
            (ttk.Button, {'text':"Full Image Comparison", 'command':self.full_image_comparison}),
            (ttk.Separator, {}),
            (ttk.Label, {'text':"Adjust Target Image:"}),
            (ttk.Label, {'text':"Brightness:"}),
            (ttk.Scale, {'from_':-100, 'to':100, 'variable':self.brightness, 'orient':'horizontal', 'command':lambda e: self.refresh_display()}),
            (ttk.Label, {'text':"Contrast:"}),
            (ttk.Scale, {'from_':0.1, 'to':3.0, 'variable':self.contrast, 'orient':'horizontal', 'command':lambda e: self.refresh_display()}),
            (ttk.Separator, {}),
            (ttk.Button, {'text':"Check Similarity", 'command':self.check_result}),
            (ttk.Button, {'text':"View Matches", 'command':self.view_matches}),
            (ttk.Button, {'text':"Save Match Image", 'command':self.save_match_image}),
            (ttk.Button, {'text':"Save Side-by-Side", 'command':self.save_side_by_side}),
            (ttk.Checkbutton, {'text':"Show Keypoints", 'variable':self.show_keypoints, 'command':self.update_keypoints}),
            (ttk.Separator, {}),
            (ttk.Label, {'text':"Algorithm:"}),
            (ttk.Combobox, {'textvariable':self.algorithm, 'values':["ORB", "SSIM", "Histogram", "Combined"], 'state':'readonly'}),
            (ttk.Label, {'text':"ORB Features:"}),
            (ttk.Scale, {'from_':100, 'to':2000, 'variable':self.orb_features, 'orient':'horizontal'}),
            (ttk.Label, {'text':"ORB Weight:"}),
            (ttk.Scale, {'from_':0.0, 'to':1.0, 'variable':self.orb_weight, 'orient':'horizontal'}),
            (ttk.Button, {'text':"Save Report", 'command':self.save_report}),
            (ttk.Button, {'text':"Reset All", 'command':self.reset_all}),
        ]
        for widget, opts in controls:
            w = widget(self.sidebar_frame, **opts)
            if isinstance(w, ttk.Separator):
                w.pack(fill='x', pady=3)
            else:
                w.pack(fill='x', pady=3)

        # ROI draw vars
        self.start_x = self.start_y = None

    def set_status(self, msg):
        self.status.config(text=msg)

    def upload_master(self):
        path = filedialog.askopenfilename(filetypes=[('Image Files','*.png;*.jpg;*.jpeg;*.bmp')])
        if path:
            self.master_img = cv2.imread(path)
            self.master_gray = cv2.cvtColor(self.master_img, cv2.COLOR_BGR2GRAY)
            self.display_image(self.master_canvas, self.master_img)
            self.set_status("Master image loaded.")

    def upload_target(self):
        path = filedialog.askopenfilename(filetypes=[('Image Files','*.png;*.jpg;*.jpeg;*.bmp')])
        if path:
            self.target_img = cv2.imread(path)
            self.target_gray = cv2.cvtColor(self.target_img, cv2.COLOR_BGR2GRAY)
            self.display_image(self.target_canvas, self.target_img)
            self.set_status("Target image loaded.")

    def apply_brightness_contrast(self, img):
        if img is None: return None
        alpha = self.contrast.get()
        beta = self.brightness.get()
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def display_image(self, canvas, img):
        disp = img.copy()
        if canvas == self.target_canvas:
            disp = self.apply_brightness_contrast(disp)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = pil.resize((canvas.winfo_width(), canvas.winfo_height()))
        tkimg = ImageTk.PhotoImage(pil)
        canvas.image = tkimg
        canvas.delete('all')
        canvas.create_image(0, 0, anchor='nw', image=tkimg)
        label = 'Master' if canvas == self.master_canvas else 'Target'
        canvas.create_text(10, 10, anchor='nw', text=label, fill='white', font=('Arial',16,'bold'))

    def refresh_display(self):
        if self.master_img is not None:
            self.display_image(self.master_canvas, self.master_img)
        if self.target_img is not None:
            self.display_image(self.target_canvas, self.target_img)

    def draw_roi_on_master(self):
        if self.master_img is None:
            messagebox.showwarning("Warning","Load master image first.")
            return
        self.master_canvas.bind("<ButtonPress-1>", self.start_roi)
        self.master_canvas.bind("<B1-Motion>", self.draw_roi)
        self.master_canvas.bind("<ButtonRelease-1>", self.save_roi)
        self.set_status("Draw ROI on master image.")

    def start_roi(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect:
            self.master_canvas.delete(self.rect)
        self.rect = self.master_canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def draw_roi(self, event):
        self.master_canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def save_roi(self, event):
        x1,y1 = self.start_x,self.start_y
        x2,y2 = event.x,event.y
        self.roi_coords = (min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2))
        self.set_status(f"ROI saved: {self.roi_coords}")

    def draw_roi_on_target(self):
        if self.roi_coords and self.target_img is not None:
            if self.target_rect:
                self.target_canvas.delete(self.target_rect)
            x1,y1,x2,y2 = self.roi_coords
            self.target_rect = self.target_canvas.create_rectangle(x1,y1,x2,y2, outline='red', width=2)
            self.set_status("ROI shown on target image.")
        else:
            messagebox.showwarning("Warning","Draw ROI on master and load target first.")

    def clear_roi(self):
        if self.rect:
            self.master_canvas.delete(self.rect)
            self.rect=None
        if self.target_rect:
            self.target_canvas.delete(self.target_rect)
            self.target_rect=None
        self.roi_coords=None
        self.set_status("ROI cleared.")

    def full_image_comparison(self):
        w,h = self.master_canvas.winfo_width(), self.master_canvas.winfo_height()
        if self.rect:
            self.master_canvas.delete(self.rect)
        self.rect = self.master_canvas.create_rectangle(0,0,w,h, outline='red', width=2)
        self.roi_coords=(0,0,w,h)
        self.draw_roi_on_target()
        self.set_status("Full image comparison selected.")

    def calculate_similarity(self):
        if not (self.master_gray is not None and self.target_gray is not None and self.roi_coords):
            return None,None,None,None
        x1,y1,x2,y2=self.roi_coords
        def map_coords(gray,x,y): h,w=gray.shape; return int(x*w/self.master_canvas.winfo_width()), int(y*h/self.master_canvas.winfo_height())
        x1m,y1m=map_coords(self.master_gray,x1,y1); x2m,y2m=map_coords(self.master_gray,x2,y2)
        x1t,y1t=map_coords(self.target_gray,x1,y1); x2t,y2t=map_coords(self.target_gray,x2,y2)
        roi_m=self.master_gray[y1m:y2m,x1m:x2m]; roi_t=self.target_gray[y1t:y2t,x1t:x2t]
        orb_score=hist_score=ssim_score=combined=0.0
        if 'ORB' in self.algorithm.get() or 'Combined' in self.algorithm.get():
            orb=cv2.ORB_create(nfeatures=self.orb_features.get())
            kp1,des1=orb.detectAndCompute(roi_m,None); kp2,des2=orb.detectAndCompute(roi_t,None)
            if des1 is not None and des2 is not None:
                bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
                matches=bf.match(des1,des2)
                orb_score=100*len(matches)/max(len(kp1),1)
        if SSIM_AVAILABLE and ('SSIM' in self.algorithm.get() or 'Combined' in self.algorithm.get()):
            try: ssim_score=ssim(roi_m,roi_t)*100
            except: ssim_score=0.0
        if 'Histogram' in self.algorithm.get():
            h1=cv2.calcHist([roi_m],[0],None,[256],[0,256]); h2=cv2.calcHist([roi_t],[0],None,[256],[0,256])
            cv2.normalize(h1,h1); cv2.normalize(h2,h2)
            corr=cv2.compareHist(h1,h2,cv2.HISTCMP_CORREL)
            hist_score=(corr+1)/2*100
        if self.algorithm.get()=='Combined':
            combined=self.orb_weight.get()*orb_score+(1-self.orb_weight.get())*ssim_score
        return orb_score,ssim_score,hist_score,combined

    def check_result(self):
        if self.master_img is None or self.target_img is None or not self.roi_coords:
            messagebox.showwarning("Warning","Please load images and define ROI.")
            return
        self.progress.start(); self.set_status("Calculating similarity..."); self.root.update()
        orb_score,ssim_score,hist_score,combined=self.calculate_similarity()
        algo=self.algorithm.get()
        if algo=='ORB': score=orb_score
        elif algo=='SSIM': score=ssim_score
        elif algo=='Histogram': score=hist_score
        else: score=combined
        self.result_label.config(text=f"Similarity ({algo}): {score:.2f}%")
        self.progress.stop(); self.set_status("Similarity calculation complete.")

    def view_matches(self):
        orb_score,_,_,_=self.calculate_similarity()
        if orb_score is None:
            messagebox.showwarning("Warning","Calculate similarity first.")
            return
        orb=cv2.ORB_create(nfeatures=self.orb_features.get())
        x1,y1,x2,y2=self.roi_coords
        def map_coords(gray,x,y): h,w=gray.shape; return int(x*w/self.master_canvas.winfo_width()), int(y*h/self.master_canvas.winfo_height())
        x1m,y1m=map_coords(self.master_gray,x1,y1); x2m,y2m=map_coords(self.master_gray,x2,y2)
        x1t,y1t=map_coords(self.target_gray,x1,y1); x2t,y2t=map_coords(self.target_gray,x2,y2)
        roi_m=self.master_gray[y1m:y2m,x1m:x2m]; roi_t=self.target_gray[y1t:y2t,x1t:x2t]
        kp1,des1=orb.detectAndCompute(roi_m,None); kp2,des2=orb.detectAndCompute(roi_t,None)
        if des1 is None or des2 is None:
            messagebox.showinfo("Info","Not enough features."); return
        bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True); matches=bf.match(des1,des2)
        matches=sorted(matches,key=lambda m: m.distance)[:20]
        match_img=cv2.drawMatches(roi_m,kp1,roi_t,kp2,matches,None,flags=2)
        self.last_match_img=match_img.copy()
        h,w=match_img.shape[:2]; max_w,max_h=1200,500
        scale=min(max_w/w,max_h/h,1.0); disp=cv2.resize(match_img,(int(w*scale),int(h*scale)))
        win=Toplevel(self.root); win.title("Top Matches")
        canvas=tk.Canvas(win,width=disp.shape[1],height=disp.shape[0]); canvas.pack()
        img=cv2.cvtColor(disp,cv2.COLOR_BGR2RGB); pil=Image.fromarray(img); tkimg=ImageTk.PhotoImage(pil)
        canvas.image=tkimg; canvas.create_image(0,0,anchor='nw',image=tkimg)
        self.set_status("Displayed top matches.")

    def save_match_image(self):
        if self.last_match_img is None:
            messagebox.showwarning("Warning","No match image.")
            return
        path=filedialog.asksaveasfilename(defaultextension='.png',filetypes=[('PNG','*.png')])
        if path:
            cv2.imwrite(path,self.last_match_img)
            self.set_status(f"Match image saved: {os.path.basename(path)}")

    def save_side_by_side(self):
        if self.master_img is None or self.target_img is None or not self.roi_coords:
            messagebox.showwarning("Warning","Load images and define ROI first.")
            return
        x1,y1,x2,y2=self.roi_coords
        h_m,w_m=self.master_img.shape[:2]
        cx1,cy1=int(x1*w_m/self.master_canvas.winfo_width()),int(y1*h_m/self.master_canvas.winfo_height())
        cx2,cy2=int(x2*w_m/self.master_canvas.winfo_width()),int(y2*h_m/self.master_canvas.winfo_height())
        roi_m=self.master_img[cy1:cy2,cx1:cx2]
        h_t,w_t=self.target_img.shape[:2]
        tx1,ty1=int(x1*w_t/self.target_canvas.winfo_width()),int(y1*h_t/self.target_canvas.winfo_height())
        tx2,ty2=int(x2*w_t/self.target_canvas.winfo_width()),int(y2*h_t/self.target_canvas.winfo_height())
        roi_t=self.target_img[ty1:ty2,tx1:tx2]
        h_new=400
        roi_m=cv2.resize(roi_m,(int(roi_m.shape[1]*h_new/roi_m.shape[0]),h_new))
        roi_t=cv2.resize(roi_t,(int(roi_t.shape[1]*h_new/roi_t.shape[0]),h_new))
        combined=np.hstack((roi_m,roi_t))
        path=filedialog.asksaveasfilename(defaultextension='.png',filetypes=[('PNG','*.png')])
        if path:
            cv2.imwrite(path,combined)
            self.set_status(f"Side-by-side saved: {os.path.basename(path)}")

    def update_keypoints(self):
        for canvas,img,gray in [(self.master_canvas,self.master_img,self.master_gray),(self.target_canvas,self.target_img,self.target_gray)]:
            if img is None or gray is None: continue
            disp=img.copy()
            if self.show_keypoints.get():
                orb=cv2.ORB_create(nfeatures=self.orb_features.get())
                kp=orb.detect(gray,None)
                disp=cv2.drawKeypoints(disp,kp,None,color=(0,255,0),flags=0)
            self.display_image(canvas,disp)
        self.set_status("Keypoints updated.")

    

    def save_report(self):
        if not self.result_label.cget('text').startswith("Similarity"): return
        path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text Files','*.txt')])
        if path:
            with open(path, 'w') as f:
                f.write(f"Report generated: {datetime.datetime.now()}\n")
                f.write(self.result_label.cget('text') + '\n')
                f.write(f"Algorithm: {self.algorithm.get()}\n")
                if self.algorithm.get() == 'Combined':
                    f.write(f"ORB Weight: {self.orb_weight.get():.2f}\n")
                f.write(f"Brightness: {self.brightness.get()}\n")
                f.write(f"Contrast: {self.contrast.get():.2f}\n")
            self.set_status(f"Report saved: {os.path.basename(path)}")

    def show_about(self):
        messagebox.showinfo("About", "Edge-based Matching App\nDeveloped with OpenCV and Tkinter\nAdvanced metrics, brightness/contrast, histogram matching, professional UI.")

    def reset_all(self):
        for c in [self.master_canvas, self.target_canvas]: c.delete('all')
        self.result_label.config(text="Similarity Score: N/A")
        self.roi_coords = None
        self.master_img = self.target_img = None
        self.master_gray = self.target_gray = None
        self.rect = self.target_rect = None
        self.last_match_img = None
        self.brightness.set(0)
        self.contrast.set(1.0)
        self.set_status("Reset complete.")

if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeComparisonApp(root)
    root.mainloop()

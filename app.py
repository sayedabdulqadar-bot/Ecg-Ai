# ecg_app.py
# Clean version (NO gloss oval). Two slides:
# Slide 1: centered heart + loading text + fill animation
# Slide 2: full report screen with ECG chart and reasons

import tkinter as tk
import math
import time
import random
from tkinter import ttk

# ---------- Geometry / heart helpers ----------
def heart_parametric_points(center_x, center_y, scale, n=400):
    pts = []
    for i in range(n):
        t = math.pi * 2 * i / n
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)
        px = center_x + x * scale
        py = center_y - y * scale
        pts.append((px, py))
    return pts

def horizontal_intersections(y, poly):
    xs = []
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        if (y1 <= y < y2) or (y2 <= y < y1):
            if y2 != y1:
                x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                xs.append(x)
    xs.sort()
    pairs = []
    for i in range(0, len(xs)-1, 2):
        pairs.append((xs[i], xs[i+1]))
    return pairs

# ---------- Synthetic ECG ----------
def generate_synthetic_ecg_points(width, height, duration_s=6, fs=200):
    samples = int(duration_s * fs)
    t = [i / fs for i in range(samples)]
    signal = []
    hr = 70
    beat_interval = 60.0/hr
    peaks = [0.5 + i*beat_interval for i in range(int(duration_s/beat_interval)+3)]
    for ti in t:
        val = 0.02*math.sin(2*math.pi*0.25*ti) + 0.01*random.uniform(-1,1)
        val += 0.05*math.sin(2*math.pi*1*ti)*0.1
        for pt in peaks:
            d = ti - pt
            if 0 <= d < 0.02:
                val += 1 * (1 - d/0.02)
        signal.append(val)

    xs = [i*(width/(len(signal)-1)) for i in range(len(signal))]
    mid = height * 0.55
    scale_y = 60
    ys = [mid - s*scale_y for s in signal]
    return list(zip(xs, ys))

# ---------- Region mapping ----------
REGION_MAP = {
    'South Asia': ['Stress', 'Obesity', 'High blood pressure', 'Advanced testing required'],
    'Southeast Asia': ['Stress', 'Obesity', 'High blood pressure', 'Tropical infections may contribute', 'Advanced testing required'],
    'Europe': ['Stress', 'Obesity', 'Hypertension', 'Advanced testing required'],
    'North America': ['Stress', 'Obesity', 'High blood pressure', 'Smoking', 'Advanced testing required'],
    'Africa': ['Infectious contributors', 'Obesity', 'High blood pressure', 'Advanced testing required'],
    'Latin America': ['Obesity', 'High blood pressure', 'Stress', 'Advanced testing required'],
    'Other/Unknown': ['Stress', 'Obesity', 'High blood pressure', 'Advanced testing required']
}

def analyze_ecg_for_demo(region):
    hr = random.randint(55, 105)
    label = "Normal"
    if hr < 50 or hr > 110:
        label = "Abnormal"
    if random.random() < 0.08:
        label = "Myocardial Infarction"
    return label, hr, REGION_MAP.get(region, REGION_MAP['Other/Unknown'])

# ---------- App ----------
class ECGApp:
    def __init__(self, root):
        self.root = root
        root.title("1-Lead ECG — First Responder")

        try:
            root.state('zoomed')
        except:
            root.attributes('-zoomed', True)

        root.bind("<Escape>", lambda e: root.destroy())

        self.w = root.winfo_screenwidth()
        self.h = root.winfo_screenheight()

        self.canvas = tk.Canvas(root, width=self.w, height=self.h,
                                bg="#0b0b0f", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Heart geometry
        self.heart_center = (self.w//2, int(self.h*0.34))
        self.heart_scale = min(self.w, self.h) / 72
        self.heart_poly = heart_parametric_points(
            self.heart_center[0], self.heart_center[1],
            self.heart_scale, n=600
        )

        self.regions = list(REGION_MAP.keys())
        self.region = tk.StringVar(value=self.regions[0])

        self.show_loading_slide()

    # -------- Slide 1 --------
    def show_loading_slide(self):
        self.canvas.delete('all')
        # Background gradient
        steps = 140
        for i in range(steps):
            r = int(12 + i * (170/steps))
            g = int(6 + i * (12/steps))
            b = int(20 + i * (40/steps))
            color = f'#{r:02x}{g:02x}{b:02x}'
            y0 = int(i*(self.h/steps))
            self.canvas.create_rectangle(0, y0, self.w, y0+(self.h//steps)+1, fill=color, outline="")

        # Heart outline
        pts = [c for p in self.heart_poly for c in p]
        self.canvas.create_polygon(pts, outline="#ffd1d9",
                                   width=6, fill="")

        # Loading text with extra spacing
        self.loading_text_id = self.canvas.create_text(
            self.w//2, int(self.h*0.62),
            text="Loading your chart readings...",
            fill="white", font=("Segoe UI", 26, "bold")
        )

        xs = [p[0] for p in self.heart_poly]
        ys = [p[1] for p in self.heart_poly]
        self.min_y, self.max_y = int(min(ys)), int(max(ys))

        self.fill_step = 0
        self.total_steps = 60
        self.delay = int(5000/self.total_steps)

        self.animate_fill_step()

    def animate_fill_step(self):
        self.canvas.delete("heart_fill")

        frac = self.fill_step / self.total_steps
        current_y = self.max_y - frac*(self.max_y - self.min_y)

        # scanlines
        max_scan = 60
        if self.max_y - int(current_y) > 0:
            dy = max(1, (self.max_y - int(current_y))//max_scan)
        else:
            dy = 1

        for yy in range(int(current_y), self.max_y+1, dy):
            for x0, x1 in horizontal_intersections(yy+0.5, self.heart_poly):
                self.canvas.create_rectangle(x0, yy, x1, yy+dy,
                                             fill="#e11b2b", outline="",
                                             tags="heart_fill")

        self.canvas.update()
        self.fill_step += 1

        if self.fill_step <= self.total_steps:
            self.canvas.after(self.delay, self.animate_fill_step)
        else:
            self.canvas.after(600, self.show_report_slide)

    # -------- Slide 2: Full Report --------
    def show_report_slide(self):
        self.canvas.delete('all')

        steps = 140
        for i in range(steps):
            r = int(20 + i*(80/steps))
            g = int(20 + i*(20/steps))
            b = int(30 + i*(80/steps))
            color = f'#{r:02x}{g:02x}{b:02x}'
            y0 = int(i*(self.h/steps))
            self.canvas.create_rectangle(0, y0, self.w, y0+(self.h//steps)+1, fill=color, outline="")

        pad = 60
        px0 = pad
        py0 = pad
        px1 = self.w - pad
        py1 = self.h - pad

        self.canvas.create_rectangle(px0, py0, px1, py1,
                                     fill="#071019", outline="#263441", width=2)

        self.canvas.create_text(self.w//2, py0+36,
                                text="ECG Reading",
                                fill="white", font=("Segoe UI", 30, "bold"))

        panel_w = px1 - px0
        panel_h = py1 - py0

        ecg_x0 = px0 + int(panel_w * 0.06)
        ecg_x1 = px1 - int(panel_w * 0.06)
        ecg_y0 = py0 + int(panel_h * 0.12)
        ecg_y1 = py0 + int(panel_h * 0.45)

        self.canvas.create_rectangle(ecg_x0, ecg_y0, ecg_x1, ecg_y1,
                                     fill="#02040a", outline="#163241")

        for gx in range(ecg_x0, ecg_x1, 40):
            self.canvas.create_line(gx, ecg_y0, gx, ecg_y1,
                                    fill="#081018")
        for gy in range(ecg_y0, ecg_y1, 20):
            self.canvas.create_line(ecg_x0, gy, ecg_x1, gy,
                                    fill="#081018")

        ecg_points = generate_synthetic_ecg_points(
            ecg_x1 - ecg_x0, ecg_y1 - ecg_y0, duration_s=6, fs=250
        )
        pts = []
        for (x, y) in ecg_points:
            pts.append(ecg_x0 + x)
            pts.append(ecg_y0 + (y - ((ecg_y1 - ecg_y0)*0.5)))

        for i in range(0, len(pts)-2, 2):
            self.canvas.create_line(pts[i], pts[i+1],
                                    pts[i+2], pts[i+3],
                                    fill="#58ff9a", width=2)

        self.class_text = self.canvas.create_text(
            self.w//2, ecg_y1+36,
            text="",
            fill="#ffefef", font=("Segoe UI", 18, "bold")
        )

        frame = tk.Frame(self.canvas, bg="#071019")
        combo = ttk.Combobox(frame, values=self.regions,
                             textvariable=self.region,
                             state="readonly", width=30)
        combo.current(0)
        combo.pack(side="left", padx=(0, 10))

        analyze_btn = tk.Button(frame, text="Analyze",
                                command=self.refresh_analysis,
                                bg="#164057", fg="white",
                                relief="flat")
        analyze_btn.pack(side="left")

        self.canvas.create_window(self.w//2, ecg_y1+86, window=frame)

        self.reasons_id = self.canvas.create_text(
            self.w//2, ecg_y1+150,
            text="",
            fill="#ffdede",
            font=("Segoe UI", 16),
            width=int(panel_w*0.7)
        )

        self.canvas.create_text(
            self.w//2, py1-30,
            text="Prototype demo — NOT medical advice. For emergencies contact emergency services.",
            fill="#97aeb6", font=("Segoe UI", 11)
        )

        self.refresh_analysis()

    def refresh_analysis(self):
        region = self.region.get()
        label, hr, reasons = analyze_ecg_for_demo(region)

        self.canvas.itemconfigure(
            self.class_text,
            text=f"Signal type: {label}    |    Heart rate: {hr} bpm"
        )

        self.canvas.itemconfigure(
            self.reasons_id,
            text="Possible reasons: " + " · ".join(reasons)
        )

# ---------- run ----------
if __name__ == "__main__":
    root = tk.Tk()
    ECGApp(root)
    root.mainloop()

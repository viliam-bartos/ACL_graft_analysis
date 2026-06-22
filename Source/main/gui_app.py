import os
import sys
import glob
import json
import threading
import traceback
import logging
from pathlib import Path

# Ujistime se, ze importujeme mri_pipeline ze stejne slozky
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

# Zabrani chybam 'NoneType' object has no attribute 'write' kdyz se pouziva pythonw.exe (sys.stdout je None)
class DummyWriter:
    def write(self, x): pass
    def flush(self): pass

if sys.stdout is None:
    sys.stdout = DummyWriter()
if sys.stderr is None:
    sys.stderr = DummyWriter()

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox  # noqa: E402
import customtkinter as ctk  # noqa: E402

class TextboxHandler(logging.Handler):
    def __init__(self, textbox):
        super().__init__()
        self.textbox = textbox

    def emit(self, record):
        # Filtr - pro GUI chceme jen INFO nebo ERROR a jen urcite formaty, 
        # nebo zkratit ty nezajimave
        msg = self.format(record)
        
        # Ignorujeme v GUI prilis detailni logy pro prehlednost
        if "Resampling" in msg or "Reorienting" in msg or "Histogram matching" in msg:
            return
            
        def append():
            self.textbox.configure(state="normal")
            self.textbox.insert(tk.END, msg + "\n")
            self.textbox.see(tk.END)
            self.textbox.configure(state="disabled")
        self.textbox.after(0, append)

# -----------------------------------------------------------------------------
# HLAVNÍ APLIKACE
# -----------------------------------------------------------------------------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ACL Analýza & Monitorování Pacienta")
        self.geometry("1100x700")
        
        ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
        
        # Nacteni konfigurace
        self.config_file = os.path.join(CURRENT_DIR, "gui_config.json")
        self.load_settings()

        # Layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.tab_process = self.tabview.add("Zpracování")
        self.tab_dashboard = self.tabview.add("Dashboard")
        self.tab_settings = self.tabview.add("Nastavení")
        
        self.build_process_tab()
        self.build_dashboard_tab()
        self.build_settings_tab()
        
        self.processing_thread = None
        
        # Nastavime GUI logovani
        self.setup_gui_logging()

    def load_settings(self):
        self.settings = {
            "anaknee_ref_mri": r"C:\ACL_analysis\ACL_graft_analysis\Data\Reference\reference.nii.gz",
            "model_ckpt": r"C:\ACL_analysis\ACL_graft_analysis\Weights\model.pth",
            "ensemble_dir": r"C:\ACL_analysis\ACL_graft_analysis\Data\5CV",
            "gt_masks_dir": r"C:\ACL_analysis\ACL_graft_analysis\Data\GT"
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.settings.update(data)
            except Exception as e:
                print("Nelze načíst config:", e)

    def save_settings(self):
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, indent=4)
            
    def setup_gui_logging(self):
        self.log_handler = TextboxHandler(self.log_textbox)
        self.log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S')
        self.log_handler.setFormatter(formatter)
        logging.getLogger().addHandler(self.log_handler)

    # -------------------------------------------------------------------------
    # 1. ZPRACOVÁNÍ TAB
    # -------------------------------------------------------------------------
    def build_process_tab(self):
        self.tab_process.grid_columnconfigure(1, weight=1)
        
        # --- Mode ---
        self.mode_var = ctk.StringVar(value="FILE")
        
        frame_mode = ctk.CTkFrame(self.tab_process)
        frame_mode.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(frame_mode, text="Mód zpracování:", font=("Arial", 14, "bold")).pack(side="left", padx=10, pady=10)
        ctk.CTkRadioButton(frame_mode, text="Jeden soubor", variable=self.mode_var, value="FILE", command=self.update_mode_ui).pack(side="left", padx=10)
        ctk.CTkRadioButton(frame_mode, text="Složka pacienta (Longitudinální)", variable=self.mode_var, value="FOLDER", command=self.update_mode_ui).pack(side="left", padx=10)

        # --- Cesty ---
        self.input_var = ctk.StringVar()
        self.output_var = ctk.StringVar()

        ctk.CTkLabel(self.tab_process, text="Vstup (Soubor/Složka):").grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.entry_input = ctk.CTkEntry(self.tab_process, textvariable=self.input_var)
        self.entry_input.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(self.tab_process, text="Procházet...", command=self.browse_input).grid(row=1, column=2, padx=10, pady=10)

        ctk.CTkLabel(self.tab_process, text="Výstupní složka:").grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.entry_output = ctk.CTkEntry(self.tab_process, textvariable=self.output_var)
        self.entry_output.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(self.tab_process, text="Procházet...", command=self.browse_output).grid(row=2, column=2, padx=10, pady=10)

        # --- Checkboxy ---
        frame_checks = ctk.CTkFrame(self.tab_process)
        frame_checks.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        self.chk_inference_var = ctk.IntVar(value=1)
        self.chk_anatomy_var = ctk.IntVar(value=1)
        self.chk_seg_var = ctk.IntVar(value=0)
        
        ctk.CTkCheckBox(frame_checks, text="Spustit Inferenci (AI segmentace)", variable=self.chk_inference_var).pack(side="left", padx=15, pady=10)
        ctk.CTkCheckBox(frame_checks, text="Anatomická analýza (Anaknee)", variable=self.chk_anatomy_var).pack(side="left", padx=15, pady=10)
        ctk.CTkCheckBox(frame_checks, text="Segmentační analýza (Porovnat s GT)", variable=self.chk_seg_var).pack(side="left", padx=15, pady=10)

        # --- Spustit ---
        self.btn_run = ctk.CTkButton(self.tab_process, text="SPUSTIT ANALÝZU", font=("Arial", 16, "bold"), fg_color="green", hover_color="darkgreen", height=40, command=self.start_processing)
        self.btn_run.grid(row=4, column=0, columnspan=3, padx=10, pady=20, sticky="ew")

        # --- Logy ---
        self.log_textbox = ctk.CTkTextbox(self.tab_process, height=200)
        self.log_textbox.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        self.log_textbox.configure(state="disabled")
        
        self.tab_process.grid_rowconfigure(5, weight=1)
        
    def update_mode_ui(self):
        self.input_var.set("")

    def browse_input(self):
        if self.mode_var.get() == "FILE":
            path = filedialog.askopenfilename(title="Vyberte MRI soubor", filetypes=[("NIfTI", "*.nii.gz *.nii")])
        else:
            path = filedialog.askdirectory(title="Vyberte složku pacienta")
        if path:
            self.input_var.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Vyberte výstupní složku")
        if path:
            self.output_var.set(path)

    def start_processing(self):
        import mri_pipeline
        
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Upozornění", "Analýza již běží!")
            return
            
        inp = self.input_var.get()
        out = self.output_var.get()
        if not inp or not out:
            messagebox.showerror("Chyba", "Vyplňte vstupní a výstupní cestu.")
            return

        self.btn_run.configure(state="disabled", text="BĚŽÍ ZPRACOVÁNÍ...")
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("0.0", "end")
        self.log_textbox.configure(state="disabled")

        # Prepni mri_pipeline nastaveni
        mri_pipeline.CONFIG["mode"] = self.mode_var.get()
        mri_pipeline.CONFIG["input_path"] = inp
        mri_pipeline.CONFIG["input_dir"] = inp
        mri_pipeline.CONFIG["output_dir"] = out
        
        mri_pipeline.CONFIG["run_inference"] = self.chk_inference_var.get()
        mri_pipeline.CONFIG["run_anatomical_analysis"] = self.chk_anatomy_var.get()
        mri_pipeline.CONFIG["run_segmentation_analysis"] = self.chk_seg_var.get()
        
        # Aplikuj skryte nastaveni
        mri_pipeline.CONFIG["anaknee_ref_mri"] = self.settings["anaknee_ref_mri"]
        mri_pipeline.CONFIG["model_ckpt"] = self.settings["model_ckpt"]
        mri_pipeline.CONFIG["ensemble_dir"] = self.settings["ensemble_dir"]
        mri_pipeline.CONFIG["gt_masks_dir"] = self.settings["gt_masks_dir"]

        self.processing_thread = threading.Thread(target=self.run_pipeline_thread)
        self.processing_thread.start()

    def run_pipeline_thread(self):
        import mri_pipeline
        import pandas as pd
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            mri_pipeline.setup_logging(mri_pipeline.CONFIG["output_dir"], "gui_pipeline.log")
            # Musime re-addovat nas GUI handler po prepisu logging configu
            logging.getLogger().addHandler(self.log_handler)
            
            logging.info(f"==== SPUŠTĚNO GUI (Zařízení: {device}) ====")
            
            # --- Modely ---
            model = None
            ensemble_models = []
            
            if mri_pipeline.CONFIG["run_inference"]:
                if mri_pipeline.CONFIG.get("use_ensemble", False):
                    logging.info("Načítám ensemble modely...")
                    ensemble_models = mri_pipeline._load_ensemble_models(mri_pipeline.CONFIG, device)
                    if not ensemble_models:
                        mri_pipeline.CONFIG["use_ensemble"] = False
                
                if not mri_pipeline.CONFIG.get("use_ensemble", False):
                    from blackwell.unets import LightUNet3D
                    model = LightUNet3D(in_ch=1, out_ch=4, base=mri_pipeline.CONFIG["base_filters"])
                    if os.path.exists(mri_pipeline.CONFIG["model_ckpt"]):
                        model.load_state_dict(torch.load(mri_pipeline.CONFIG["model_ckpt"], map_location=device))
                        model.to(device)
                        logging.info("Single model načten.")
                    else:
                        logging.warning("Model nenalezen!")
                        model = None

            # --- Zpracování ---
            results_list = []
            
            if mri_pipeline.CONFIG["mode"] == "FILE":
                res = mri_pipeline.process_single_volume(mri_pipeline.CONFIG["input_path"], model, device, run_viz_at_end=False, ensemble_models=ensemble_models)
                if res:
                    results_list.append(res)
                
            elif mri_pipeline.CONFIG["mode"] == "FOLDER":
                files = sorted(glob.glob(os.path.join(mri_pipeline.CONFIG["input_dir"], "*.nii*")))
                logging.info(f"Nalezeno {len(files)} souborů ve složce.")
                for f in files:
                    res = mri_pipeline.process_single_volume(f, model, device, run_viz_at_end=False, ensemble_models=ensemble_models)
                    if res:
                        results_list.append(res)
                
                if mri_pipeline.CONFIG["run_segmentation_analysis"]:
                    mri_pipeline.perform_segmentation_analysis(mri_pipeline.CONFIG["output_dir"], mri_pipeline.CONFIG["gt_masks_dir"])

            # --- Ulozeni pacientovych vysledku ---
            if results_list:
                df = pd.DataFrame(results_list)
                csv_path = os.path.join(mri_pipeline.CONFIG["output_dir"], "vysledky_pacienta.csv")
                
                # Pokud uz existuje, updatneme ho
                if os.path.exists(csv_path):
                    df_old = pd.read_csv(csv_path)
                    df_combined = pd.concat([df_old, df]).drop_duplicates(subset=["Filename"], keep="last")
                    df_combined.to_csv(csv_path, index=False)
                else:
                    df.to_csv(csv_path, index=False)
                    
                logging.info(f"Uloženy souhrnné výsledky do: {csv_path}")

            logging.info("==== HOTOVO ====")
            
            # Po dokonceni refreshneme dashboard
            csv_path = os.path.join(mri_pipeline.CONFIG["output_dir"], "vysledky_pacienta.csv")
            if os.path.exists(csv_path):
                self.after(0, lambda: self.load_dashboard_data(csv_path))

        except Exception as e:
            logging.error(f"Fatální chyba v pipeline: {e}")
            traceback.print_exc()
        finally:
            self.after(0, lambda: self.btn_run.configure(state="normal", text="SPUSTIT ANALÝZU"))

    # -------------------------------------------------------------------------
    # 2. DASHBOARD TAB
    # -------------------------------------------------------------------------
    def build_dashboard_tab(self):
        self.tab_dashboard.grid_columnconfigure(0, weight=1)
        self.tab_dashboard.grid_rowconfigure(1, weight=1)
        
        # --- Top controls ---
        frame_controls = ctk.CTkFrame(self.tab_dashboard)
        frame_controls.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkButton(frame_controls, text="Načíst vysledky_pacienta.csv", command=self.browse_csv).pack(side="left", padx=10, pady=10)
        
        ctk.CTkLabel(frame_controls, text="Zobrazit graf pro:").pack(side="left", padx=10)
        self.metric_var = ctk.StringVar(value="ATT_mm")
        self.opt_metric = ctk.CTkOptionMenu(frame_controls, variable=self.metric_var, values=["ATT_mm", "acl_volume_mm3", "Tortuosity_Index", "notch_width_mm", "Staubli_Tibial_pct", "BH_Length_pct", "BH_Depth_pct"], command=self.update_graph)
        self.opt_metric.pack(side="left", padx=10)
        
        # --- Graph Area ---
        self.frame_graph = ctk.CTkFrame(self.tab_dashboard)
        self.frame_graph.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # --- 3D Vizualizace složka ---
        frame_viz = ctk.CTkFrame(self.tab_dashboard)
        frame_viz.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(frame_viz, text="Skeny pacienta (pro 3D náhled):").pack(side="left", padx=10)
        self.viz_scan_var = ctk.StringVar()
        self.opt_scan = ctk.CTkOptionMenu(frame_viz, variable=self.viz_scan_var, values=[])
        self.opt_scan.pack(side="left", padx=10)
        
        ctk.CTkButton(frame_viz, text="Zobrazit 3D", command=self.show_3d).pack(side="left", padx=10)

        self.df_patient = None
        self.current_csv_dir = ""

    def browse_csv(self):
        path = filedialog.askopenfilename(title="Vyberte CSV s výsledky pacienta", filetypes=[("CSV", "*.csv")])
        if path:
            self.load_dashboard_data(path)
            
    def load_dashboard_data(self, csv_path):
        import pandas as pd
        
        if not os.path.exists(csv_path):
            return
            
        try:
            self.df_patient = pd.read_csv(csv_path)
            self.current_csv_dir = os.path.dirname(csv_path)
            
            # Naplneni option menu se skeny
            if "Filename" in self.df_patient.columns:
                scans = self.df_patient["Filename"].tolist()
                self.opt_scan.configure(values=scans)
                if scans:
                    self.viz_scan_var.set(scans[-1])
            
            self.update_graph()
        except Exception as e:
            messagebox.showerror("Chyba", f"Nepodařilo se načíst data: {e}")

    def update_graph(self, *args):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        if self.df_patient is None or self.df_patient.empty:
            return
            
        metric = self.metric_var.get()
        if metric not in self.df_patient.columns:
            return

        # Smazat predchozi graf
        for widget in self.frame_graph.winfo_children():
            widget.destroy()
            
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#2b2b2b")
        ax.set_facecolor("#2b2b2b")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
            
        # Vytvorit x-osu bud z casu nebo z indexu
        x_labels = self.df_patient["Filename"].apply(lambda x: x.split('_')[0][:10]) if "Filename" in self.df_patient.columns else self.df_patient.index
        
        ax.plot(x_labels, self.df_patient[metric], marker='o', color='cyan', linestyle='-', linewidth=2, markersize=8)
        ax.set_title(f"Vývoj: {metric}", color="white", fontsize=14)
        ax.grid(True, color="#444444", linestyle='--')
        
        fig.autofmt_xdate()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.frame_graph)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_3d(self):
        import mri_pipeline
        
        filename = self.viz_scan_var.get()
        if not filename or not self.current_csv_dir:
            return
            
        # Pokus o nalezení zdrojového obrazu automaticky
        img_path = os.path.join(mri_pipeline.CONFIG.get("input_dir", ""), filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.current_csv_dir, filename)
            
        # Pokud neexistuje automaticky na nalezené cestě, vyzveme uživatele
        if not os.path.exists(img_path):
            messagebox.showinfo("Výběr vstupního obrazu", f"Původní obraz '{filename}' nebyl automaticky nalezen.\nProsím, vyberte ho ručně.")
            img_path = filedialog.askopenfilename(title=f"Vyberte původní sken {filename}", filetypes=[("NIfTI", "*.nii.gz *.nii")])
            if not img_path:
                return # Zrušeno uživatelem
                
        mask_basename = filename if filename.startswith("mask_") else f"mask_{filename}"
        mask_path = os.path.join(self.current_csv_dir, mask_basename)
        
        if not os.path.exists(mask_path):
            messagebox.showerror("Chyba", f"Nepodařilo se najít masku ve výsledcích.\nHledáno:\nMaska: {mask_path}")
            return
            
        ref_path = self.settings["anaknee_ref_mri"]
        
        # Spustime v threadu aby nezamrzlo GUI
        def worker():
            mri_pipeline.run_visualization_only(img_path, ref_path, mask_path)
            
        threading.Thread(target=worker).start()

    # -------------------------------------------------------------------------
    # 3. NASTAVENÍ TAB
    # -------------------------------------------------------------------------
    def build_settings_tab(self):
        self.tab_settings.grid_columnconfigure(1, weight=1)
        
        self.sv_ref = ctk.StringVar(value=self.settings.get("anaknee_ref_mri", ""))
        self.sv_model = ctk.StringVar(value=self.settings.get("model_ckpt", ""))
        self.sv_ens = ctk.StringVar(value=self.settings.get("ensemble_dir", ""))
        self.sv_gt = ctk.StringVar(value=self.settings.get("gt_masks_dir", ""))
        
        self.add_setting_row("Referenční MRI (Anaknee):", self.sv_ref, 0)
        self.add_setting_row("Váhy modelu (single):", self.sv_model, 1)
        self.add_setting_row("Složka Ensemble modelů:", self.sv_ens, 2, is_dir=True)
        self.add_setting_row("Složka GT masek:", self.sv_gt, 3, is_dir=True)
        
        ctk.CTkButton(self.tab_settings, text="ULOŽIT NASTAVENÍ", command=self.save_settings_from_ui).grid(row=4, column=0, columnspan=3, pady=20)

    def add_setting_row(self, label, string_var, row, is_dir=False):
        ctk.CTkLabel(self.tab_settings, text=label).grid(row=row, column=0, padx=10, pady=10, sticky="e")
        ctk.CTkEntry(self.tab_settings, textvariable=string_var).grid(row=row, column=1, padx=10, pady=10, sticky="ew")
        
        def browse():
            if is_dir:
                path = filedialog.askdirectory()
            else:
                path = filedialog.askopenfilename()
            if path:
                string_var.set(path)
                
        ctk.CTkButton(self.tab_settings, text="Procházet...", command=browse).grid(row=row, column=2, padx=10, pady=10)

    def save_settings_from_ui(self):
        self.settings["anaknee_ref_mri"] = self.sv_ref.get()
        self.settings["model_ckpt"] = self.sv_model.get()
        self.settings["ensemble_dir"] = self.sv_ens.get()
        self.settings["gt_masks_dir"] = self.sv_gt.get()
        self.save_settings()
        messagebox.showinfo("Uloženo", "Nastavení bylo uloženo.")

if __name__ == "__main__":
    app = App()
    app.mainloop()

import os
import sys
import glob
import json
import threading
import traceback
import logging
import subprocess
from pathlib import Path

# Ensure we import mri_pipeline from the current folder
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))


class DummyWriter:
    def write(self, x):
        pass

    def flush(self):
        pass


if sys.stdout is None:
    sys.stdout = DummyWriter()
if sys.stderr is None:
    sys.stderr = DummyWriter()

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox  # noqa: E402
import customtkinter as ctk  # noqa: E402


class TextboxHandler(logging.Handler):
    """Routes log messages to the GUI textbox."""

    def __init__(self, textbox):
        super().__init__()
        self.textbox = textbox

    def emit(self, record):
        msg = self.format(record)

        # Filter out overly detailed logs for the GUI
        if "Resampling" in msg or "Reorienting" in msg or "Histogram matching" in msg:
            return

        def append():
            self.textbox.configure(state="normal")
            self.textbox.insert(tk.END, msg + "\n")
            self.textbox.see(tk.END)
            self.textbox.configure(state="disabled")

        self.textbox.after(0, append)


# ── Color Palette ───────────────────────────────────────────────────
ACCENT = "#1abc9c"
ACCENT_HOVER = "#16a085"
SUCCESS = "#27ae60"
SUCCESS_HOVER = "#219a52"


# ── Main Application ────────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ACL Graft Analysis")
        self.geometry("1100x750")
        self.minsize(900, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Fonts
        self.font_title = ctk.CTkFont(family="Segoe UI", size=15, weight="bold")
        self.font_body = ctk.CTkFont(family="Segoe UI", size=13)
        self.font_small = ctk.CTkFont(family="Segoe UI", size=11)
        self.font_button = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        self.font_run = ctk.CTkFont(family="Segoe UI", size=16, weight="bold")

        # Load GUI configuration
        self.config_file = os.path.join(CURRENT_DIR, "gui_config.json")
        self.load_settings()

        # Layout setup
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        self.tab_process = self.tabview.add("Processing")
        self.tab_dashboard = self.tabview.add("Dashboard")
        self.tab_settings = self.tabview.add("Settings")

        self.build_process_tab()
        self.build_dashboard_tab()
        self.build_settings_tab()

        self.processing_thread = None
        self.setup_gui_logging()

    # ── Config Persistence ──────────────────────────────────────────

    def load_settings(self):
        self.settings = {
            "anaknee_ref_mri": r"Data\Reference\reference.nii.gz",
            "model_ckpt": r"Weights\model.pth",
            "ensemble_dir": r"Data\5CV",
            "gt_masks_dir": r"Data\GT",
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.settings.update(data)
            except Exception as e:
                logging.warning(f"Could not load config: {e}")

    def save_settings(self):
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            logging.warning(f"Could not save config: {e}")

    def setup_gui_logging(self):
        self.log_handler = TextboxHandler(self.log_textbox)
        self.log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
        self.log_handler.setFormatter(formatter)
        logging.getLogger().addHandler(self.log_handler)

    # ── Processing Tab ──────────────────────────────────────────────

    def build_process_tab(self):
        self.tab_process.grid_columnconfigure(0, weight=1)

        # Mode selection
        frame_mode = ctk.CTkFrame(self.tab_process, corner_radius=8)
        frame_mode.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")

        self.mode_var = ctk.StringVar(value="FILE")

        ctk.CTkLabel(
            frame_mode, text="Mode:", font=self.font_title
        ).pack(side="left", padx=(15, 10), pady=10)
        ctk.CTkRadioButton(
            frame_mode, text="Single File", variable=self.mode_var,
            value="FILE", font=self.font_body, command=self.update_mode_ui,
        ).pack(side="left", padx=10)
        ctk.CTkRadioButton(
            frame_mode, text="Patient Folder", variable=self.mode_var,
            value="FOLDER", font=self.font_body, command=self.update_mode_ui,
        ).pack(side="left", padx=10)

        # Input / Output paths
        frame_paths = ctk.CTkFrame(self.tab_process, corner_radius=8)
        frame_paths.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        frame_paths.grid_columnconfigure(1, weight=1)

        self.input_var = ctk.StringVar()
        self.output_var = ctk.StringVar()

        ctk.CTkLabel(
            frame_paths, text="Input:", font=self.font_body
        ).grid(row=0, column=0, padx=(15, 5), pady=8, sticky="e")
        ctk.CTkEntry(
            frame_paths, textvariable=self.input_var, font=self.font_small,
            placeholder_text="NIfTI file, DICOM file, or folder...",
        ).grid(row=0, column=1, padx=5, pady=8, sticky="ew")
        ctk.CTkButton(
            frame_paths, text="Browse", width=80, font=self.font_body,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.browse_input,
        ).grid(row=0, column=2, padx=(5, 15), pady=8)

        ctk.CTkLabel(
            frame_paths, text="Output:", font=self.font_body
        ).grid(row=1, column=0, padx=(15, 5), pady=8, sticky="e")
        ctk.CTkEntry(
            frame_paths, textvariable=self.output_var, font=self.font_small,
            placeholder_text="Auto-filled when input is selected...",
        ).grid(row=1, column=1, padx=5, pady=8, sticky="ew")
        ctk.CTkButton(
            frame_paths, text="Browse", width=80, font=self.font_body,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.browse_output,
        ).grid(row=1, column=2, padx=(5, 15), pady=8)

        # Module toggles
        frame_modules = ctk.CTkFrame(self.tab_process, corner_radius=8)
        frame_modules.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(
            frame_modules, text="Modules:", font=self.font_title
        ).pack(side="left", padx=(15, 10), pady=10)

        self.chk_inference_var = ctk.IntVar(value=1)
        self.chk_anatomy_var = ctk.IntVar(value=1)
        self.chk_seg_var = ctk.IntVar(value=0)

        ctk.CTkCheckBox(
            frame_modules, text="AI Segmentation",
            variable=self.chk_inference_var, font=self.font_body,
        ).pack(side="left", padx=12, pady=10)
        ctk.CTkCheckBox(
            frame_modules, text="Anatomical Analysis",
            variable=self.chk_anatomy_var, font=self.font_body,
        ).pack(side="left", padx=12, pady=10)
        ctk.CTkCheckBox(
            frame_modules, text="Compare with GT",
            variable=self.chk_seg_var, font=self.font_body,
        ).pack(side="left", padx=12, pady=10)

        # Run button + progress bar
        frame_run = ctk.CTkFrame(
            self.tab_process, corner_radius=8, fg_color="transparent"
        )
        frame_run.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        frame_run.grid_columnconfigure(0, weight=1)

        self.btn_run = ctk.CTkButton(
            frame_run, text="\u25b6  RUN ANALYSIS", font=self.font_run,
            fg_color=SUCCESS, hover_color=SUCCESS_HOVER, height=45,
            corner_radius=8, command=self.start_processing,
        )
        self.btn_run.grid(row=0, column=0, padx=0, pady=(5, 2), sticky="ew")

        self.progress = ctk.CTkProgressBar(frame_run, height=4, corner_radius=2)
        self.progress.grid(row=1, column=0, padx=0, pady=(0, 5), sticky="ew")
        self.progress.set(0)

        # Log textbox
        self.log_textbox = ctk.CTkTextbox(
            self.tab_process, height=200, font=self.font_small,
            corner_radius=8, state="disabled",
        )
        self.log_textbox.grid(row=4, column=0, padx=10, pady=(5, 10), sticky="nsew")

        self.tab_process.grid_rowconfigure(4, weight=1)

    def update_mode_ui(self):
        self.input_var.set("")
        self.output_var.set("")

    def browse_input(self):
        if self.mode_var.get() == "FILE":
            path = filedialog.askopenfilename(
                title="Select MRI file",
                filetypes=[
                    ("All Supported", "*.nii.gz *.nii *.dcm"),
                    ("NIfTI", "*.nii.gz *.nii"),
                    ("DICOM", "*.dcm"),
                ],
            )
        else:
            path = filedialog.askdirectory(title="Select patient folder")
        if path:
            self.input_var.set(path)
            self._auto_suggest_output(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    def _auto_suggest_output(self, input_path):
        """Auto-fill output folder when input is selected and output is empty."""
        if self.output_var.get():
            return
        if os.path.isfile(input_path):
            parent = os.path.dirname(input_path)
        else:
            parent = input_path
        self.output_var.set(os.path.join(parent, "Results"))

    # ── Laterality Prompt ───────────────────────────────────────────

    def ask_laterality(self, filename):
        """Thread-safe laterality prompt called from the worker thread."""
        result = {"value": "Left"}
        event = threading.Event()

        def show_dialog():
            dialog = ctk.CTkToplevel(self)
            dialog.title("Knee Laterality")
            dialog.geometry("420x220")
            dialog.resizable(False, False)
            dialog.transient(self)
            dialog.grab_set()
            dialog.lift()
            dialog.focus_force()

            # Center on parent window
            self.update_idletasks()
            x = self.winfo_x() + (self.winfo_width() - 420) // 2
            y = self.winfo_y() + (self.winfo_height() - 220) // 2
            dialog.geometry(f"+{x}+{y}")

            ctk.CTkLabel(
                dialog, text="Laterality Not Detected",
                font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            ).pack(pady=(20, 5))

            ctk.CTkLabel(
                dialog,
                text=f"File: {filename}\n\nIs this a left or right knee?",
                font=ctk.CTkFont(family="Segoe UI", size=13),
                wraplength=380,
            ).pack(pady=(5, 15))

            btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            btn_frame.pack(pady=10)

            def choose(val):
                result["value"] = val
                event.set()
                dialog.destroy()

            ctk.CTkButton(
                btn_frame, text="\u2b05  Left Knee", width=140, height=38,
                font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                fg_color=ACCENT, hover_color=ACCENT_HOVER,
                command=lambda: choose("Left"),
            ).pack(side="left", padx=10)
            ctk.CTkButton(
                btn_frame, text="Right Knee  \u27a1", width=140, height=38,
                font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                fg_color=ACCENT, hover_color=ACCENT_HOVER,
                command=lambda: choose("Right"),
            ).pack(side="left", padx=10)

            dialog.protocol("WM_DELETE_WINDOW", lambda: choose("Left"))

        self.after(0, show_dialog)
        event.wait()
        return result["value"]

    # ── Processing Logic ────────────────────────────────────────────

    def start_processing(self):
        try:
            import mri_pipeline  # noqa: F401
        except ImportError as e:
            messagebox.showerror(
                "Import Error", f"Could not load pipeline module:\n{e}"
            )
            return

        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Warning", "Analysis is already running!")
            return

        inp = self.input_var.get()
        out = self.output_var.get()
        if not inp:
            messagebox.showerror("Error", "Please select an input file or folder.")
            return
        if not out:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        self.btn_run.configure(state="disabled", text="\u23f3  PROCESSING...")
        self.progress.configure(mode="indeterminate")
        self.progress.start()

        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("0.0", "end")
        self.log_textbox.configure(state="disabled")

        # Set pipeline config
        import mri_pipeline

        mri_pipeline.CONFIG["mode"] = self.mode_var.get()
        mri_pipeline.CONFIG["input_path"] = inp
        mri_pipeline.CONFIG["input_dir"] = inp
        mri_pipeline.CONFIG["output_dir"] = out

        mri_pipeline.CONFIG["run_inference"] = self.chk_inference_var.get()
        mri_pipeline.CONFIG["run_anatomical_analysis"] = self.chk_anatomy_var.get()
        mri_pipeline.CONFIG["run_segmentation_analysis"] = self.chk_seg_var.get()

        # Apply settings
        mri_pipeline.CONFIG["anaknee_ref_mri"] = self.settings["anaknee_ref_mri"]
        mri_pipeline.CONFIG["model_ckpt"] = self.settings["model_ckpt"]
        mri_pipeline.CONFIG["ensemble_dir"] = self.settings["ensemble_dir"]
        mri_pipeline.CONFIG["gt_masks_dir"] = self.settings["gt_masks_dir"]

        self.processing_thread = threading.Thread(
            target=self.run_pipeline_thread, daemon=True
        )
        self.processing_thread.start()

    def run_pipeline_thread(self):
        import mri_pipeline

        try:
            import pandas as pd
            import torch
        except ImportError as e:
            self.after(
                0,
                lambda: messagebox.showerror(
                    "Import Error", f"Missing dependency:\n{e}"
                ),
            )
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            mri_pipeline.setup_logging(
                mri_pipeline.CONFIG["output_dir"], "pipeline.log"
            )
            # Re-add log handler after setup_logging config override
            logging.getLogger().addHandler(self.log_handler)

            logging.info(f"==== Analysis started (Device: {device}) ====")

            model = None
            ensemble_models = []

            if mri_pipeline.CONFIG["run_inference"]:
                if mri_pipeline.CONFIG.get("use_ensemble", False):
                    logging.info("Loading ensemble models...")
                    ensemble_models = mri_pipeline._load_ensemble_models(
                        mri_pipeline.CONFIG, device
                    )
                    if not ensemble_models:
                        mri_pipeline.CONFIG["use_ensemble"] = False

                if not mri_pipeline.CONFIG.get("use_ensemble", False):
                    try:
                        from blackwell.unets import LightUNet3D

                        model = LightUNet3D(
                            in_ch=1, out_ch=4,
                            base=mri_pipeline.CONFIG["base_filters"],
                        )
                        ckpt = mri_pipeline.CONFIG["model_ckpt"]
                        if os.path.exists(ckpt):
                            model.load_state_dict(
                                torch.load(ckpt, map_location=device)
                            )
                            model.to(device)
                            logging.info("Single model loaded.")
                        else:
                            logging.warning(f"Model checkpoint not found: {ckpt}")
                            model = None
                    except Exception as e:
                        logging.error(f"Could not load model: {e}")
                        model = None

            inp = mri_pipeline.CONFIG["input_path"]
            out = mri_pipeline.CONFIG["output_dir"]
            results_list = []

            # Detect DICOM input and convert if needed
            files_to_process = []

            if mri_pipeline.CONFIG["mode"] == "FILE":
                if mri_pipeline.is_dicom_input(inp):
                    logging.info("DICOM input detected, converting to NIfTI...")
                    nifti_paths = mri_pipeline.convert_dicom_to_nifti(inp, out)
                    files_to_process = nifti_paths
                else:
                    files_to_process = [inp]

            elif mri_pipeline.CONFIG["mode"] == "FOLDER":
                if mri_pipeline.is_dicom_input(inp):
                    logging.info("DICOM folder detected, converting to NIfTI...")
                    nifti_paths = mri_pipeline.convert_dicom_to_nifti(inp, out)
                    files_to_process = nifti_paths
                else:
                    files_to_process = sorted(
                        glob.glob(os.path.join(inp, "*.nii*"))
                    )
                    logging.info(f"Found {len(files_to_process)} NIfTI files.")

            # Process each file
            for f in files_to_process:
                try:
                    res = mri_pipeline.process_single_volume(
                        f, model, device,
                        run_viz_at_end=False,
                        ensemble_models=ensemble_models,
                        laterality_callback=self.ask_laterality,
                    )
                    if res:
                        results_list.append(res)
                except Exception as e:
                    logging.error(f"Error processing {os.path.basename(f)}: {e}")
                    traceback.print_exc()

            if (
                mri_pipeline.CONFIG["mode"] == "FOLDER"
                and mri_pipeline.CONFIG["run_segmentation_analysis"]
            ):
                mri_pipeline.perform_segmentation_analysis(
                    out, mri_pipeline.CONFIG["gt_masks_dir"]
                )

            # Save results
            csv_path = os.path.join(out, "patient_results.csv")
            if results_list:
                df = pd.DataFrame(results_list)
                if os.path.exists(csv_path):
                    df_old = pd.read_csv(csv_path)
                    df = pd.concat([df_old, df]).drop_duplicates(
                        subset=["Filename"], keep="last"
                    )
                df.to_csv(csv_path, index=False)
                logging.info(f"Results saved to: {csv_path}")

            logging.info("==== Analysis complete ====")

            # Refresh dashboard
            if os.path.exists(csv_path):
                self.after(0, lambda: self.load_dashboard_data(csv_path))

            # Show completion summary
            n = len(results_list)
            self.after(0, lambda: self.show_completion_summary(n, out))

        except Exception as e:
            logging.error(f"Pipeline error: {e}")
            traceback.print_exc()
            self.after(
                0,
                lambda: messagebox.showerror("Error", f"Pipeline failed:\n{e}"),
            )
        finally:

            def reset_ui():
                self.btn_run.configure(
                    state="normal", text="\u25b6  RUN ANALYSIS"
                )
                self.progress.stop()
                self.progress.set(0)

            self.after(0, reset_ui)

    # ── Completion Summary ──────────────────────────────────────────

    def show_completion_summary(self, num_files, output_dir):
        """Show a summary dialog after processing completes."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Analysis Complete")
        dialog.geometry("450x250")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        dialog.lift()

        # Center on parent window
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 450) // 2
        y = self.winfo_y() + (self.winfo_height() - 250) // 2
        dialog.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            dialog, text="\u2705  Analysis Complete",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
        ).pack(pady=(25, 10))

        ctk.CTkLabel(
            dialog,
            text=f"{num_files} file(s) processed successfully.",
            font=ctk.CTkFont(family="Segoe UI", size=13),
        ).pack(pady=5)

        ctk.CTkLabel(
            dialog,
            text=f"\U0001f4c1  {output_dir}",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="gray60",
            wraplength=400,
        ).pack(pady=5)

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=15)

        ctk.CTkButton(
            btn_frame, text="Open Folder", width=130,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            command=lambda: self._open_folder(output_dir),
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            btn_frame, text="Close", width=100,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            fg_color="gray30", hover_color="gray40",
            command=dialog.destroy,
        ).pack(side="left", padx=8)

    @staticmethod
    def _open_folder(path):
        """Open folder in system file explorer."""
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass

    # ── Dashboard Tab ───────────────────────────────────────────────

    def build_dashboard_tab(self):
        self.tab_dashboard.grid_columnconfigure(0, weight=1)
        self.tab_dashboard.grid_rowconfigure(1, weight=1)

        # Top controls
        frame_controls = ctk.CTkFrame(self.tab_dashboard, corner_radius=8)
        frame_controls.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ctk.CTkButton(
            frame_controls, text="Load Results CSV", font=self.font_body,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.browse_csv,
        ).pack(side="left", padx=10, pady=10)

        ctk.CTkLabel(
            frame_controls, text="Metric:", font=self.font_body
        ).pack(side="left", padx=(15, 5))

        self.metric_var = ctk.StringVar(value="ATT_mm")
        self.opt_metric = ctk.CTkOptionMenu(
            frame_controls, variable=self.metric_var,
            values=[
                "ATT_mm", "acl_volume_mm3", "Tortuosity_Index",
                "notch_width_mm", "Staubli_Tibial_pct",
                "BH_Length_pct", "BH_Depth_pct",
            ],
            font=self.font_small, command=self.update_graph,
        )
        self.opt_metric.pack(side="left", padx=5)

        ctk.CTkButton(
            frame_controls, text="\U0001f4c2 Open Folder", font=self.font_body,
            fg_color="gray30", hover_color="gray40",
            command=self._open_results_folder,
        ).pack(side="right", padx=10, pady=10)

        # Graph area
        self.frame_graph = ctk.CTkFrame(self.tab_dashboard, corner_radius=8)
        self.frame_graph.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # 3D visualization section
        frame_viz = ctk.CTkFrame(self.tab_dashboard, corner_radius=8)
        frame_viz.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")

        ctk.CTkLabel(
            frame_viz, text="3D Preview:", font=self.font_body
        ).pack(side="left", padx=(15, 5), pady=10)

        self.viz_scan_var = ctk.StringVar()
        self.opt_scan = ctk.CTkOptionMenu(
            frame_viz, variable=self.viz_scan_var, values=[],
            font=self.font_small,
        )
        self.opt_scan.pack(side="left", padx=5)

        ctk.CTkButton(
            frame_viz, text="Show 3D", font=self.font_body,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self.show_3d,
        ).pack(side="left", padx=10)

        self.df_patient = None
        self.current_csv_dir = ""

    def _open_results_folder(self):
        if self.current_csv_dir and os.path.isdir(self.current_csv_dir):
            self._open_folder(self.current_csv_dir)
        else:
            messagebox.showinfo("Info", "No results loaded yet.")

    def browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select patient results CSV",
            filetypes=[("CSV", "*.csv")],
        )
        if path:
            self.load_dashboard_data(path)

    def load_dashboard_data(self, csv_path):
        try:
            import pandas as pd
        except ImportError:
            messagebox.showerror("Error", "pandas is required for the dashboard.")
            return

        if not os.path.exists(csv_path):
            return

        try:
            self.df_patient = pd.read_csv(csv_path)
            self.current_csv_dir = os.path.dirname(csv_path)

            # Populate scan option menu
            if "Filename" in self.df_patient.columns:
                scans = self.df_patient["Filename"].tolist()
                self.opt_scan.configure(values=scans)
                if scans:
                    self.viz_scan_var.set(scans[-1])

            self.update_graph()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load data:\n{e}")

    def update_graph(self, *args):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError:
            return

        if self.df_patient is None or self.df_patient.empty:
            return

        metric = self.metric_var.get()
        if metric not in self.df_patient.columns:
            return

        # Clear previous plot
        for widget in self.frame_graph.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#333355")

        # Generate X labels from filenames or index
        x_labels = (
            self.df_patient["Filename"].apply(
                lambda x: str(x).split("_")[0][:12]
            )
            if "Filename" in self.df_patient.columns
            else self.df_patient.index
        )

        ax.plot(
            x_labels, self.df_patient[metric],
            marker="o", color=ACCENT, linestyle="-",
            linewidth=2, markersize=8,
            markerfacecolor=ACCENT_HOVER,
            markeredgecolor="white", markeredgewidth=1.5,
        )
        ax.set_title(
            f"Trend: {metric}", color="white", fontsize=13, fontweight="bold"
        )
        ax.set_ylabel(metric, color="gray", fontsize=10)
        ax.grid(True, color="#333355", linestyle="--", alpha=0.5)

        fig.autofmt_xdate()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.frame_graph)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_3d(self):
        try:
            import mri_pipeline
        except ImportError:
            messagebox.showerror("Error", "Pipeline module not available.")
            return

        filename = self.viz_scan_var.get()
        if not filename or not self.current_csv_dir:
            messagebox.showinfo("Info", "Select a scan first.")
            return

        # Attempt to locate source image
        img_path = os.path.join(
            mri_pipeline.CONFIG.get("input_dir", ""), filename
        )
        if not os.path.exists(img_path):
            img_path = os.path.join(self.current_csv_dir, filename)

        # Fallback to manual selection if not found
        if not os.path.exists(img_path):
            messagebox.showinfo(
                "File Not Found",
                f"Original scan '{filename}' not found.\n"
                "Please select it manually.",
            )
            img_path = filedialog.askopenfilename(
                title=f"Select scan: {filename}",
                filetypes=[("NIfTI", "*.nii.gz *.nii")],
            )
            if not img_path:
                return

        mask_basename = (
            filename if filename.startswith("mask_") else f"mask_{filename}"
        )
        mask_path = os.path.join(self.current_csv_dir, mask_basename)

        if not os.path.exists(mask_path):
            messagebox.showerror("Error", f"Mask not found:\n{mask_path}")
            return

        ref_path = self.settings["anaknee_ref_mri"]

        # Run visualization in worker thread to prevent GUI freezing
        def worker():
            try:
                mri_pipeline.run_visualization_only(
                    img_path, ref_path, mask_path
                )
            except Exception as e:
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        "3D Error", f"Could not open 3D viewer:\n{e}"
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    # ── Settings Tab ────────────────────────────────────────────────

    def build_settings_tab(self):
        self.tab_settings.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self.tab_settings, text="Pipeline Configuration",
            font=self.font_title,
        ).grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w")

        self.sv_ref = ctk.StringVar(
            value=self.settings.get("anaknee_ref_mri", "")
        )
        self.sv_model = ctk.StringVar(
            value=self.settings.get("model_ckpt", "")
        )
        self.sv_ens = ctk.StringVar(
            value=self.settings.get("ensemble_dir", "")
        )
        self.sv_gt = ctk.StringVar(
            value=self.settings.get("gt_masks_dir", "")
        )

        self._add_setting_row("Reference MRI:", self.sv_ref, 1)
        self._add_setting_row("Model Weights:", self.sv_model, 2)
        self._add_setting_row("Ensemble Folder:", self.sv_ens, 3, is_dir=True)
        self._add_setting_row("GT Masks Folder:", self.sv_gt, 4, is_dir=True)

        ctk.CTkButton(
            self.tab_settings, text="\U0001f4be  Save Settings",
            font=self.font_button, fg_color=SUCCESS, hover_color=SUCCESS_HOVER,
            height=38, corner_radius=8, command=self.save_settings_from_ui,
        ).grid(row=5, column=0, columnspan=3, padx=15, pady=20, sticky="ew")

    def _add_setting_row(self, label, string_var, row, is_dir=False):
        ctk.CTkLabel(
            self.tab_settings, text=label, font=self.font_body,
        ).grid(row=row, column=0, padx=(15, 5), pady=8, sticky="e")

        ctk.CTkEntry(
            self.tab_settings, textvariable=string_var, font=self.font_small,
        ).grid(row=row, column=1, padx=5, pady=8, sticky="ew")

        def browse():
            if is_dir:
                path = filedialog.askdirectory()
            else:
                path = filedialog.askopenfilename()
            if path:
                string_var.set(path)

        ctk.CTkButton(
            self.tab_settings, text="Browse", width=80, font=self.font_body,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, command=browse,
        ).grid(row=row, column=2, padx=(5, 15), pady=8)

    def save_settings_from_ui(self):
        self.settings["anaknee_ref_mri"] = self.sv_ref.get()
        self.settings["model_ckpt"] = self.sv_model.get()
        self.settings["ensemble_dir"] = self.sv_ens.get()
        self.settings["gt_masks_dir"] = self.sv_gt.get()
        self.save_settings()
        messagebox.showinfo("Saved", "Settings have been saved successfully.")


if __name__ == "__main__":
    app = App()
    app.mainloop()

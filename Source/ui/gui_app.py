import os
import sys
import glob
import json
import threading
import traceback
import logging
import subprocess
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CURRENT_DIR.parent
ROOT_DIR = SOURCE_DIR.parent

for p in [str(CURRENT_DIR), str(SOURCE_DIR), str(SOURCE_DIR / "pipeline"), str(SOURCE_DIR / "anaknee")]:
    if p not in sys.path:
        sys.path.insert(0, p)

class DummyWriter:
    def write(self, x): pass
    def flush(self): pass

if sys.stdout is None:
    sys.stdout = DummyWriter()
if sys.stderr is None:
    sys.stderr = DummyWriter()

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# ── Color Palette (Medical Dark Cyberpunk / Modern Clinical) ─────────
ACCENT_CYAN = "#06b6d4"       # Primary 3D & Action
ACCENT_CYAN_HOVER = "#0891b2"
ACCENT_GREEN = "#10b981"      # Run & Success
ACCENT_GREEN_HOVER = "#059669"
BG_CARD = "#1e293b"           # Slate-800
BG_CARD_LIGHT = "#334155"     # Slate-700
BG_INPUT = "#0f172a"          # Slate-900
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"
BORDER_COLOR = "#475569"


class TextboxHandler(logging.Handler):
    """Logging handler that writes to a CTk textbox in a thread-safe manner."""
    def __init__(self, textbox):
        super().__init__()
        self.textbox = textbox

    def emit(self, record):
        msg = self.format(record)

        def append():
            try:
                self.textbox.configure(state="normal")
                self.textbox.insert(tk.END, msg + "\n")
                self.textbox.see(tk.END)
                self.textbox.configure(state="disabled")
            except Exception:
                pass

        self.textbox.after(0, append)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ACL Graft Analysis & 3D Geometric Visualizer")
        self.geometry("1180x820")
        self.minsize(980, 680)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Typography
        self.font_h1 = ctk.CTkFont(family="Segoe UI", size=18, weight="bold")
        self.font_title = ctk.CTkFont(family="Segoe UI", size=14, weight="bold")
        self.font_body = ctk.CTkFont(family="Segoe UI", size=12)
        self.font_body_bold = ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        self.font_small = ctk.CTkFont(family="Segoe UI", size=11)
        self.font_mono = ctk.CTkFont(family="Consolas", size=11)
        self.font_btn = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        self.font_btn_lg = ctk.CTkFont(family="Segoe UI", size=15, weight="bold")

        self.config_file = os.path.join(CURRENT_DIR, "gui_config.json")
        self.load_settings()

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.build_header()

        # Tabview
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")

        # Order: 3D Viewer FIRST (user priority), then Pipeline, Results, Settings
        self.tab_viewer = self.tabview.add("🚀 3D Prohlížeč (PyVista)")
        self.tab_process = self.tabview.add("⚡ Dávková Analýza")
        self.tab_dashboard = self.tabview.add("📊 Výsledky & Případy")
        self.tab_settings = self.tabview.add("⚙ Nastavení")

        self.build_viewer_tab()
        self.build_process_tab()
        self.build_dashboard_tab()
        self.build_settings_tab()

        self.processing_thread = None
        self.setup_gui_logging()

        # Pre-warm modules in background so there is zero import delay on click!
        threading.Thread(target=self._prewarm_engines, daemon=True).start()

        # Auto-check reference data on launch
        self.after(300, self._auto_detect_project_data)

    def _prewarm_engines(self):
        """Asynchronously load visualizer and analysis modules in memory at startup."""
        try:
            import SimpleITK
            import pyvista
            from anaknee import visualizator_analyzator
            from anaknee import main_acl_analysis
            self.after(0, lambda: self.lbl_global_status.configure(text="● Připraveno (Bleskový režim aktivní)", text_color=ACCENT_GREEN))
        except Exception as e:
            logging.debug(f"Pre-warm note: {e}")

    def load_settings(self):
        self.settings = {
            "anaknee_ref_mri": os.path.join(str(ROOT_DIR), "Data", "reference", "right_case_074.nii.gz"),
            "model_ckpt": os.path.join(str(ROOT_DIR), "Weights", "model.pth"),
            "ensemble_dir": os.path.join(str(ROOT_DIR), "Data", "5CV"),
            "gt_masks_dir": os.path.join(str(ROOT_DIR), "Data", "GT"),
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

    # ── Top Header Bar ──────────────────────────────────────────────
    def build_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent", height=45)
        header.grid(row=0, column=0, padx=20, pady=(12, 6), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            title_frame, text="🦵 ACL GRAFT ANALYSIS",
            font=self.font_h1, text_color=TEXT_MAIN
        ).pack(side="left", padx=(0, 10))

        ctk.CTkLabel(
            title_frame, text="|  3D Segmentace & Biomechanika Kolene",
            font=self.font_body, text_color=TEXT_MUTED
        ).pack(side="left")

        # Right status badge
        self.lbl_global_status = ctk.CTkLabel(
            header, text="● Připraveno",
            font=self.font_small, text_color=ACCENT_GREEN
        )
        self.lbl_global_status.grid(row=0, column=1, sticky="e", padx=5)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1: 🚀 3D PROHLÍŽEČ (PYVISTA) — HLAVNÍ POŽADAVEK UŽIVATELE
    # ═══════════════════════════════════════════════════════════════════
    def build_viewer_tab(self):
        self.tab_viewer.grid_columnconfigure(0, weight=1)
        self.tab_viewer.grid_rowconfigure(2, weight=1)

        # ── Karta 1: Výběr objemu ───────────────────────────────────
        card_sel = ctk.CTkFrame(self.tab_viewer, fg_color=BG_CARD, corner_radius=10)
        card_sel.grid(row=0, column=0, padx=12, pady=(10, 6), sticky="ew")
        card_sel.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            card_sel, text="📂 Výběr objemu k 3D zobrazení",
            font=self.font_title, text_color=ACCENT_CYAN
        ).grid(row=0, column=0, columnspan=3, padx=15, pady=(12, 8), sticky="w")

        # Primární soubor (Maska nebo MRI sken)
        self.viz_primary_var = ctk.StringVar()
        self.viz_secondary_var = ctk.StringVar()

        ctk.CTkLabel(
            card_sel, text="Hlavní objem (NIfTI / DICOM):", font=self.font_body_bold
        ).grid(row=1, column=0, padx=(15, 8), pady=6, sticky="e")

        self.entry_primary = ctk.CTkEntry(
            card_sel, textvariable=self.viz_primary_var, font=self.font_small,
            fg_color=BG_INPUT, border_color=BORDER_COLOR,
            placeholder_text="Vyberte segmentační masku (mask_*.nii.gz) nebo libovolný MRI sken (.nii, .nii.gz, .dcm)..."
        )
        self.entry_primary.grid(row=1, column=1, padx=6, pady=6, sticky="ew")

        ctk.CTkButton(
            card_sel, text="Procházet...", width=95, font=self.font_body,
            fg_color=ACCENT_CYAN, hover_color=ACCENT_CYAN_HOVER,
            command=self._browse_viz_primary
        ).grid(row=1, column=2, padx=(6, 15), pady=6)

        # Volitelný doplňkový soubor (kontext)
        ctk.CTkLabel(
            card_sel, text="Doplňková maska / MRI (volitelné):", font=self.font_body
        ).grid(row=2, column=0, padx=(15, 8), pady=6, sticky="e")

        self.entry_secondary = ctk.CTkEntry(
            card_sel, textvariable=self.viz_secondary_var, font=self.font_small,
            fg_color=BG_INPUT, border_color=BORDER_COLOR,
            placeholder_text="Volitelné: doplňkový MRI sken k masce nebo maska k MRI..."
        )
        self.entry_secondary.grid(row=2, column=1, padx=6, pady=6, sticky="ew")

        ctk.CTkButton(
            card_sel, text="Procházet...", width=95, font=self.font_body,
            fg_color="gray30", hover_color="gray40",
            command=self._browse_viz_secondary
        ).grid(row=2, column=2, padx=(6, 15), pady=6)

        # ── Rychlé předvolby (1-klik načtení) ────────────────────────
        preset_frame = ctk.CTkFrame(card_sel, fg_color="transparent")
        preset_frame.grid(row=3, column=0, columnspan=3, padx=15, pady=(8, 12), sticky="w")

        ctk.CTkLabel(preset_frame, text="⚡ Rychlé načtení z projektu:", font=self.font_small, text_color=TEXT_MUTED).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            preset_frame, text="📁 Referenční maska (074)", font=self.font_small, height=28,
            fg_color=BG_CARD_LIGHT, hover_color=BORDER_COLOR,
            command=self._load_preset_mask
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            preset_frame, text="📁 Referenční MRI sken", font=self.font_small, height=28,
            fg_color=BG_CARD_LIGHT, hover_color=BORDER_COLOR,
            command=self._load_preset_mri
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            preset_frame, text="📁 Poslední výsledek", font=self.font_small, height=28,
            fg_color=BG_CARD_LIGHT, hover_color=BORDER_COLOR,
            command=self._load_preset_latest
        ).pack(side="left", padx=4)

        # ── Karta 2: Detekce & Spuštění ──────────────────────────────
        card_launch = ctk.CTkFrame(self.tab_viewer, fg_color=BG_CARD, corner_radius=10)
        card_launch.grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        card_launch.grid_columnconfigure(0, weight=1)

        # Dynamický detekční banner
        self.lbl_detection_badge = ctk.CTkLabel(
            card_launch,
            text="ℹ Zadejte nebo vyberte soubor výše pro okamžité 3D zobrazení.",
            font=self.font_body, text_color=TEXT_MUTED, anchor="w", justify="left"
        )
        self.lbl_detection_badge.grid(row=0, column=0, padx=15, pady=(12, 6), sticky="ew")

        # Volba rychlého režimu
        self.chk_fast_mode_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            card_launch,
            text="Bleskový režim geometrie (výpočet za ~1,5 s, přeskočí zdlouhavou radiomiku)",
            variable=self.chk_fast_mode_var,
            font=self.font_small, text_color=TEXT_MAIN
        ).grid(row=1, column=0, padx=15, pady=4, sticky="w")

        # Hlavní spouštěcí tlačítko
        btn_action_frame = ctk.CTkFrame(card_launch, fg_color="transparent")
        btn_action_frame.grid(row=2, column=0, padx=15, pady=(10, 14), sticky="ew")
        btn_action_frame.grid_columnconfigure(0, weight=1)

        self.btn_launch_3d = ctk.CTkButton(
            btn_action_frame, text="▶  OTEVŘÍT V PYVISTA 3D", font=self.font_btn_lg,
            fg_color=ACCENT_CYAN, hover_color=ACCENT_CYAN_HOVER, height=48,
            corner_radius=8, command=self.action_launch_pyvista
        )
        self.btn_launch_3d.grid(row=0, column=0, sticky="ew")

        # ── Karta 3: Přehled metrik po zobrazení ──────────────────────
        self.card_metrics_preview = ctk.CTkFrame(self.tab_viewer, fg_color=BG_CARD, corner_radius=10)
        self.card_metrics_preview.grid(row=2, column=0, padx=12, pady=(6, 12), sticky="nsew")
        self.card_metrics_preview.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(
            self.card_metrics_preview, text="📊 Rychlý náhled biomechanických parametrů",
            font=self.font_title, text_color=TEXT_MAIN
        ).grid(row=0, column=0, columnspan=4, padx=15, pady=(12, 6), sticky="w")

        # 4 boxíky metrik
        self.metric_boxes = {}
        metric_defs = [
            ("att", "ATT (mm)", "0.0 mm", "Přední posun tibie"),
            ("staubli", "Stäubli Tibia", "0.0 %", "AP pozice úponu (norma 40-44%)"),
            ("elevation", "Úhel plata", "0.0°", "Sklon vazu k tibiálnímu platu"),
            ("bh_len", "B&H Délka", "0.0 %", "Femorální úpon v mřížce"),
        ]
        for col, (m_id, label, default_val, desc) in enumerate(metric_defs):
            b_frame = ctk.CTkFrame(self.card_metrics_preview, fg_color=BG_INPUT, corner_radius=8)
            b_frame.grid(row=1, column=col, padx=8, pady=8, sticky="nsew")
            ctk.CTkLabel(b_frame, text=label, font=self.font_small, text_color=TEXT_MUTED).pack(pady=(6, 2))
            val_lbl = ctk.CTkLabel(b_frame, text=default_val, font=self.font_h1, text_color=ACCENT_CYAN)
            val_lbl.pack(pady=(0, 2))
            ctk.CTkLabel(b_frame, text=desc, font=ctk.CTkFont(size=9), text_color="gray50").pack(pady=(0, 6))
            self.metric_boxes[m_id] = val_lbl

        self.lbl_viewer_status = ctk.CTkLabel(
            self.card_metrics_preview,
            text="Prohlížeč připraven. Po otevření PyVista se okno zobrazí v plném 3D s interaktivními vrstvami.",
            font=self.font_small, text_color=TEXT_MUTED
        )
        self.lbl_viewer_status.grid(row=2, column=0, columnspan=4, padx=15, pady=8, sticky="w")

    def _browse_viz_primary(self):
        path = filedialog.askopenfilename(
            title="Vyberte objem (Maska nebo MRI sken)",
            filetypes=[
                ("Všechny podporované", "*.nii.gz *.nii *.dcm"),
                ("NIfTI soubory", "*.nii.gz *.nii"),
                ("DICOM", "*.dcm"),
            ],
        )
        if path:
            self.viz_primary_var.set(path)
            self._update_detection_info(path)

    def _browse_viz_secondary(self):
        path = filedialog.askopenfilename(
            title="Vyberte doplňkový soubor (maska nebo sken)",
            filetypes=[
                ("NIfTI soubory", "*.nii.gz *.nii"),
                ("Všechny soubory", "*.*"),
            ],
        )
        if path:
            self.viz_secondary_var.set(path)

    def _auto_detect_project_data(self):
        """Preload paths if reference files exist in repository."""
        ref_mask = os.path.join(ROOT_DIR, "Data", "reference", "vysledky_074", "mask_right_case_074.nii.gz")
        if os.path.exists(ref_mask) and not self.viz_primary_var.get():
            self.viz_primary_var.set(ref_mask)
            self._update_detection_info(ref_mask)

    def _load_preset_mask(self):
        p = os.path.join(ROOT_DIR, "Data", "reference", "vysledky_074", "mask_right_case_074.nii.gz")
        if os.path.exists(p):
            self.viz_primary_var.set(p)
            self._update_detection_info(p)
        else:
            messagebox.showinfo("Informace", f"Referenční maska nebyla nalezena v:\n{p}")

    def _load_preset_mri(self):
        p = os.path.join(ROOT_DIR, "Data", "reference", "right_case_074.nii.gz")
        if os.path.exists(p):
            self.viz_primary_var.set(p)
            self._update_detection_info(p)
        else:
            messagebox.showinfo("Informace", f"Referenční MRI sken nebyl nalezen v:\n{p}")

    def _load_preset_latest(self):
        # Look in Data/reference/Results or output folders
        candidates = glob.glob(os.path.join(ROOT_DIR, "Data", "**", "mask_*.nii*"), recursive=True)
        if candidates:
            latest = sorted(candidates, key=os.path.getmtime)[-1]
            self.viz_primary_var.set(latest)
            self._update_detection_info(latest)
        else:
            messagebox.showinfo("Informace", "Zatím nebyly nalezeny žádné vytvořené masky.")

    def _update_detection_info(self, path):
        fname = os.path.basename(path).lower()
        if "mask" in fname:
            self.lbl_detection_badge.configure(
                text="🟢 Detekována SEGMENTAČNÍ MASKA (Femur, Tibia, ACL) → Plná 3D anatomie s RANSAC platem a úhly.",
                text_color=ACCENT_GREEN
            )
        else:
            self.lbl_detection_badge.configure(
                text="🔵 Detekován MRI OBJEM (Intenzitní sken) → 3D ortogonální řezy (Axial, Coronal, Sagittal) v PyVista.",
                text_color=ACCENT_CYAN
            )

    def action_launch_pyvista(self):
        primary = self.viz_primary_var.get().strip()
        secondary = self.viz_secondary_var.get().strip()

        if not primary or not os.path.exists(primary):
            messagebox.showerror("Chyba", "Vyberte prosím platný soubor k zobrazení.")
            return

        self.btn_launch_3d.configure(state="disabled", text="⏳  ZPRACOVÁVÁM...")
        self.lbl_viewer_status.configure(text=f"⏳ [1/4] Spouštím analýzu: {os.path.basename(primary)}...", text_color=ACCENT_CYAN)
        self.update_idletasks()

        def worker():
            try:
                from anaknee.visualizator_analyzator import smart_visualize, visualize_results, visualize_mri_volume
                import SimpleITK as sitk
                import numpy as np

                fname = os.path.basename(primary).lower()
                is_mask = "mask" in fname

                def on_progress(step_text):
                    self.after(0, lambda: self.lbl_viewer_status.configure(text=f"⏳ {step_text}", text_color=ACCENT_CYAN))

                if is_mask or (secondary and "mask" in secondary.lower()):
                    mask_to_use = primary if is_mask else secondary
                    from anaknee.main_acl_analysis import run_geometric_analysis_from_mask
                    res_dict, mask_array, spacing_zyx, f_cent, t_cent, p_info, vis_data = run_geometric_analysis_from_mask(
                        mask_to_use, progress_callback=on_progress
                    )

                    on_progress("[4/4] Generování 3D polygonálních sítí v PyVista...")

                    # Update metric cards
                    def update_ui_metrics():
                        self.metric_boxes["att"].configure(text=f"{res_dict.get('ATT_mm', 0):.2f} mm")
                        self.metric_boxes["staubli"].configure(text=f"{res_dict.get('Staubli_Tibial_pct', 0):.1f} %")
                        self.metric_boxes["elevation"].configure(text=f"{res_dict.get('angle_to_plateau_deg', 0):.1f}°")
                        self.metric_boxes["bh_len"].configure(text=f"{res_dict.get('BH_Length_pct', 0):.1f} %")
                        self.lbl_viewer_status.configure(
                            text=f"✓ 3D model otevřen! [Inliers plata: {len(vis_data.get('plateau_inliers') or [])}]",
                            text_color=ACCENT_GREEN
                        )
                    self.after(0, update_ui_metrics)

                    # Open PyVista
                    visualize_results(mask_array, spacing_zyx, vis_data)
                else:
                    # Grayscale MRI volume
                    on_progress("Načítám 3D ortogonální řezy MRI skenu...")
                    def update_ui_vol():
                        self.lbl_viewer_status.configure(
                            text=f"✓ 3D MRI Prohlížeč řezů otevřen pro {os.path.basename(primary)}.",
                            text_color=ACCENT_CYAN
                        )
                    self.after(0, update_ui_vol)
                    visualize_mri_volume(primary)

            except Exception as e:
                traceback.print_exc()
                self.after(0, lambda: messagebox.showerror("Chyba vizualizace", f"Nepodařilo se spustit 3D zobrazení:\n{e}"))
            finally:
                self.after(0, lambda: self.btn_launch_3d.configure(state="normal", text="▶  OTEVŘÍT V PYVISTA 3D"))

        threading.Thread(target=worker, daemon=True).start()

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2: ⚡ DÁVKOVÁ ANALÝZA (PIPELINE)
    # ═══════════════════════════════════════════════════════════════════
    def build_process_tab(self):
        self.tab_process.grid_columnconfigure(0, weight=1)

        # Karta: Vstup a výstup
        card_io = ctk.CTkFrame(self.tab_process, fg_color=BG_CARD, corner_radius=10)
        card_io.grid(row=0, column=0, padx=12, pady=(10, 6), sticky="ew")
        card_io.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(card_io, text="📁 Vstupní a výstupní cesty", font=self.font_title, text_color=TEXT_MAIN).grid(row=0, column=0, columnspan=3, padx=15, pady=(12, 6), sticky="w")

        # Mode radio buttons
        self.mode_var = ctk.StringVar(value="FILE")
        mode_frame = ctk.CTkFrame(card_io, fg_color="transparent")
        mode_frame.grid(row=1, column=0, columnspan=3, padx=15, pady=4, sticky="w")
        ctk.CTkLabel(mode_frame, text="Režim:", font=self.font_body_bold).pack(side="left", padx=(0, 10))
        ctk.CTkRadioButton(mode_frame, text="Jednotlivý soubor", variable=self.mode_var, value="FILE", font=self.font_body).pack(side="left", padx=10)
        ctk.CTkRadioButton(mode_frame, text="Složka pacienta (dávka / DICOM)", variable=self.mode_var, value="FOLDER", font=self.font_body).pack(side="left", padx=10)

        # Input / Output entries
        self.input_var = ctk.StringVar()
        self.output_var = ctk.StringVar()

        ctk.CTkLabel(card_io, text="Vstup:", font=self.font_body).grid(row=2, column=0, padx=(15, 6), pady=6, sticky="e")
        ctk.CTkEntry(card_io, textvariable=self.input_var, font=self.font_small, fg_color=BG_INPUT, border_color=BORDER_COLOR, placeholder_text="Cesta k NIfTI nebo DICOM souboru/složce...").grid(row=2, column=1, padx=6, pady=6, sticky="ew")
        ctk.CTkButton(card_io, text="Procházet...", width=95, font=self.font_body, fg_color=ACCENT_CYAN, hover_color=ACCENT_CYAN_HOVER, command=self.browse_input).grid(row=2, column=2, padx=(6, 15), pady=6)

        ctk.CTkLabel(card_io, text="Výstup:", font=self.font_body).grid(row=3, column=0, padx=(15, 6), pady=6, sticky="e")
        ctk.CTkEntry(card_io, textvariable=self.output_var, font=self.font_small, fg_color=BG_INPUT, border_color=BORDER_COLOR, placeholder_text="Cesta k výsledné složce (např. Data/Results)...").grid(row=3, column=1, padx=6, pady=6, sticky="ew")
        ctk.CTkButton(card_io, text="Procházet...", width=95, font=self.font_body, fg_color="gray30", hover_color="gray40", command=self.browse_output).grid(row=3, column=2, padx=(6, 15), pady=(6, 14))

        # Karta: Moduly analýzy
        card_mod = ctk.CTkFrame(self.tab_process, fg_color=BG_CARD, corner_radius=10)
        card_mod.grid(row=1, column=0, padx=12, pady=6, sticky="ew")

        ctk.CTkLabel(card_mod, text="⚙ Volby pipeline", font=self.font_title, text_color=TEXT_MAIN).pack(anchor="w", padx=15, pady=(10, 4))

        opt_frame = ctk.CTkFrame(card_mod, fg_color="transparent")
        opt_frame.pack(fill="x", padx=15, pady=(0, 10))

        self.chk_inference_var = ctk.IntVar(value=1)
        self.chk_anatomy_var = ctk.IntVar(value=1)
        self.chk_radiomics_var = ctk.IntVar(value=0)  # Off by default for 10x speedup!
        self.chk_autoviz_var = ctk.IntVar(value=1)

        ctk.CTkCheckBox(opt_frame, text="AI Segmentace (LightUNet3D)", variable=self.chk_inference_var, font=self.font_body).pack(side="left", padx=10, pady=4)
        ctk.CTkCheckBox(opt_frame, text="Geometrická analýza (Anaknee)", variable=self.chk_anatomy_var, font=self.font_body).pack(side="left", padx=10, pady=4)
        ctk.CTkCheckBox(opt_frame, text="PyRadiomics (pomalé)", variable=self.chk_radiomics_var, font=self.font_body).pack(side="left", padx=10, pady=4)
        ctk.CTkCheckBox(opt_frame, text="Otevřít 3D po dokončení", variable=self.chk_autoviz_var, font=self.font_body).pack(side="left", padx=10, pady=4)

        # Spouštěcí karta
        card_run = ctk.CTkFrame(self.tab_process, fg_color="transparent")
        card_run.grid(row=2, column=0, padx=12, pady=6, sticky="ew")
        card_run.grid_columnconfigure(0, weight=1)

        self.btn_run = ctk.CTkButton(
            card_run, text="▶  SPUSTIT ANALÝZU PIPELINE", font=self.font_btn_lg,
            fg_color=ACCENT_GREEN, hover_color=ACCENT_GREEN_HOVER, height=46,
            corner_radius=8, command=self.start_processing
        )
        self.btn_run.grid(row=0, column=0, sticky="ew", pady=(2, 4))

        self.progress = ctk.CTkProgressBar(card_run, height=6, corner_radius=3)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        self.progress.set(0)

        # Log textbox
        self.log_textbox = ctk.CTkTextbox(
            self.tab_process, height=180, font=self.font_mono,
            fg_color=BG_CARD, corner_radius=8, state="disabled"
        )
        self.log_textbox.grid(row=3, column=0, padx=12, pady=(6, 12), sticky="nsew")
        self.tab_process.grid_rowconfigure(3, weight=1)

    def browse_input(self):
        if self.mode_var.get() == "FILE":
            path = filedialog.askopenfilename(
                title="Vyberte MRI soubor",
                filetypes=[
                    ("Všechny podporované", "*.nii.gz *.nii *.dcm"),
                    ("NIfTI", "*.nii.gz *.nii"),
                    ("DICOM", "*.dcm"),
                ],
            )
        else:
            path = filedialog.askdirectory(title="Vyberte složku pacienta / DICOM sérii")
        if path:
            self.input_var.set(path)
            self._auto_suggest_output(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Vyberte výstupní složku")
        if path:
            self.output_var.set(path)

    def _auto_suggest_output(self, input_path):
        if self.output_var.get():
            return
        parent = os.path.dirname(input_path) if os.path.isfile(input_path) else input_path
        self.output_var.set(os.path.join(parent, "Results"))

    def start_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Upozornění", "Analýza již běží!")
            return

        inp = self.input_var.get().strip()
        out = self.output_var.get().strip()
        if not inp:
            messagebox.showerror("Chyba", "Vyberte prosím vstupní soubor nebo složku.")
            return
        if not out:
            messagebox.showerror("Chyba", "Vyberte prosím výstupní složku.")
            return

        self.btn_run.configure(state="disabled", text="⏳  ZPRACOVÁVÁM...")
        self.progress.configure(mode="indeterminate")
        self.progress.start()

        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("0.0", "end")
        self.log_textbox.configure(state="disabled")

        self._thread_config = {
            "mode": self.mode_var.get(),
            "input_path": inp,
            "input_dir": inp,
            "output_dir": out,
            "run_inference": self.chk_inference_var.get(),
            "run_anatomical_analysis": self.chk_anatomy_var.get(),
            "compute_radiomics": bool(self.chk_radiomics_var.get()),
            "anaknee_ref_mri": self.settings["anaknee_ref_mri"],
            "model_ckpt": self.settings["model_ckpt"],
            "ensemble_dir": self.settings["ensemble_dir"],
            "gt_masks_dir": self.settings["gt_masks_dir"],
        }

        self.processing_thread = threading.Thread(target=self.run_pipeline_thread, daemon=True)
        self.processing_thread.start()

    def run_pipeline_thread(self):
        try:
            import mri_pipeline
            import pandas as pd
            import torch
        except ImportError as e:
            self.after(0, lambda: messagebox.showerror("Chyba importu", f"Chybí závislost:\n{e}"))
            return

        for key, val in self._thread_config.items():
            mri_pipeline.CONFIG[key] = val

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            mri_pipeline.setup_logging(mri_pipeline.CONFIG["output_dir"], "pipeline.log")
            logging.getLogger().addHandler(self.log_handler)
            logging.info(f"==== Analýza spuštěna (Zařízení: {device}) ====")

            model = None
            ensemble_models = []

            if mri_pipeline.CONFIG["run_inference"]:
                if mri_pipeline.CONFIG.get("use_ensemble", True):
                    logging.info("Načítám 5-Fold ansámbl modelů...")
                    ensemble_models = mri_pipeline._load_ensemble_models(mri_pipeline.CONFIG, device)

            inp = mri_pipeline.CONFIG["input_path"]
            out = mri_pipeline.CONFIG["output_dir"]
            os.makedirs(out, exist_ok=True)
            results_list = []

            if mri_pipeline.CONFIG["mode"] == "FILE":
                if mri_pipeline.is_dicom_input(inp):
                    files_to_process = mri_pipeline.convert_dicom_to_nifti(inp, out)
                else:
                    files_to_process = [inp]
            else:
                files_to_process = sorted(glob.glob(os.path.join(inp, "*.nii*")))

            for f in files_to_process:
                try:
                    res = mri_pipeline.process_single_volume(
                        f, model, device,
                        run_viz_at_end=False,
                        ensemble_models=ensemble_models,
                        laterality_callback=lambda fn: "Right" if "right" in fn.lower() else "Left",
                    )
                    if res:
                        results_list.append(res)
                except Exception as e:
                    logging.error(f"Chyba při zpracování {os.path.basename(f)}: {e}")
                    traceback.print_exc()

            csv_path = os.path.join(out, "patient_results.csv")
            if results_list:
                df = pd.DataFrame(results_list)
                if os.path.exists(csv_path):
                    df_old = pd.read_csv(csv_path)
                    df = pd.concat([df_old, df]).drop_duplicates(subset=["Filename"], keep="last")
                df.to_csv(csv_path, index=False)
                logging.info(f"Výsledky uloženy: {csv_path}")

            logging.info("==== Analýza dokončena ====")
            if os.path.exists(csv_path):
                self.after(0, lambda: self.load_dashboard_data(csv_path))

            # Auto-viz if requested
            if self.chk_autoviz_var.get() and files_to_process:
                first_file = files_to_process[0]
                base_name = os.path.basename(first_file)
                mask_path = os.path.join(out, f"mask_{base_name}")
                if os.path.exists(mask_path):
                    self.after(500, lambda mp=mask_path: self._auto_open_viz(mp))

        except Exception as e:
            logging.error(f"Chyba pipeline: {e}")
            self.after(0, lambda: messagebox.showerror("Chyba", f"Chyba pipeline:\n{e}"))
        finally:
            def reset_ui():
                self.btn_run.configure(state="normal", text="▶  SPUSTIT ANALÝZU PIPELINE")
                self.progress.stop()
                self.progress.set(0)
            self.after(0, reset_ui)

    def _auto_open_viz(self, mask_path):
        self.viz_primary_var.set(mask_path)
        self._update_detection_info(mask_path)
        self.tabview.set("🚀 3D Prohlížeč (PyVista)")
        self.action_launch_pyvista()

    # ═══════════════════════════════════════════════════════════════════
    # TAB 3: 📊 VÝSLEDKY & PŘÍPADY (DASHBOARD)
    # ═══════════════════════════════════════════════════════════════════
    def build_dashboard_tab(self):
        self.tab_dashboard.grid_columnconfigure(0, weight=1)
        self.tab_dashboard.grid_rowconfigure(1, weight=1)

        # Controls bar
        ctrl_frame = ctk.CTkFrame(self.tab_dashboard, fg_color=BG_CARD, corner_radius=10)
        ctrl_frame.grid(row=0, column=0, padx=12, pady=(10, 6), sticky="ew")

        ctk.CTkButton(
            ctrl_frame, text="Načíst CSV výsledků", font=self.font_btn,
            fg_color=ACCENT_CYAN, hover_color=ACCENT_CYAN_HOVER,
            command=self.browse_csv
        ).pack(side="left", padx=12, pady=10)

        ctk.CTkLabel(ctrl_frame, text="Zobrazit metriku v grafu:", font=self.font_body).pack(side="left", padx=(15, 6))

        self.metric_var = ctk.StringVar(value="ATT_mm")
        self.opt_metric = ctk.CTkOptionMenu(
            ctrl_frame, variable=self.metric_var,
            values=["ATT_mm", "Staubli_Tibial_pct", "BH_Length_pct", "BH_Depth_pct", "angle_to_plateau_deg", "acl_volume_mm3", "Tortuosity_Index", "notch_width_mm"],
            font=self.font_small, command=self.update_graph,
        )
        self.opt_metric.pack(side="left", padx=6)

        ctk.CTkButton(
            ctrl_frame, text="📂 Otevřít složku", font=self.font_body,
            fg_color=BG_CARD_LIGHT, hover_color=BORDER_COLOR,
            command=self._open_results_folder
        ).pack(side="right", padx=12, pady=10)

        # Content: Split into Table of cases and Graph
        content_frame = ctk.CTkFrame(self.tab_dashboard, fg_color="transparent")
        content_frame.grid(row=1, column=0, padx=12, pady=6, sticky="nsew")
        content_frame.grid_columnconfigure(0, weight=3)
        content_frame.grid_columnconfigure(1, weight=2)
        content_frame.grid_rowconfigure(0, weight=1)

        # Left: Table of cases with 1-click 3D buttons
        table_frame = ctk.CTkFrame(content_frame, fg_color=BG_CARD, corner_radius=10)
        table_frame.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        table_frame.grid_rowconfigure(1, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(table_frame, text="📋 Zpracované případy (Klikněte na 👁 3D pro okamžité otevření)", font=self.font_title, text_color=TEXT_MAIN).grid(row=0, column=0, padx=15, pady=(10, 6), sticky="w")

        self.cases_scroll = ctk.CTkScrollableFrame(table_frame, fg_color="transparent")
        self.cases_scroll.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

        # Right: Trend graph
        self.frame_graph = ctk.CTkFrame(content_frame, fg_color=BG_CARD, corner_radius=10)
        self.frame_graph.grid(row=0, column=1, padx=(6, 0), sticky="nsew")

        self.df_patient = None
        self.current_csv_dir = ""

    def _open_results_folder(self):
        if self.current_csv_dir and os.path.isdir(self.current_csv_dir):
            if sys.platform == "win32":
                os.startfile(self.current_csv_dir)
            else:
                subprocess.Popen(["xdg-open", self.current_csv_dir])
        else:
            messagebox.showinfo("Informace", "Zatím není načtena žádná složka s výsledky.")

    def browse_csv(self):
        path = filedialog.askopenfilename(
            title="Vyberte CSV soubor s výsledky",
            filetypes=[("CSV soubory", "*.csv")],
        )
        if path:
            self.load_dashboard_data(path)

    def load_dashboard_data(self, csv_path):
        try:
            import pandas as pd
            self.df_patient = pd.read_csv(csv_path)
            self.current_csv_dir = os.path.dirname(csv_path)

            # Clear existing table
            for widget in self.cases_scroll.winfo_children():
                widget.destroy()

            # Populate table rows
            for idx, row in self.df_patient.iterrows():
                fname = str(row.get("Filename", f"Case_{idx}"))
                att = row.get("ATT_mm", 0)
                stb = row.get("Staubli_Tibial_pct", 0)
                bh = row.get("BH_Length_pct", 0)

                row_frame = ctk.CTkFrame(self.cases_scroll, fg_color=BG_INPUT, corner_radius=6)
                row_frame.pack(fill="x", pady=4, padx=2)

                ctk.CTkLabel(row_frame, text=fname[:24], font=self.font_body_bold, width=170, anchor="w").pack(side="left", padx=8, pady=6)
                ctk.CTkLabel(row_frame, text=f"ATT: {att:.1f} mm", font=self.font_small, text_color=ACCENT_CYAN, width=90).pack(side="left", padx=4)
                ctk.CTkLabel(row_frame, text=f"Stäubli: {stb:.1f}%", font=self.font_small, text_color=TEXT_MUTED, width=90).pack(side="left", padx=4)

                # Direct 1-click 3D button!
                btn_3d = ctk.CTkButton(
                    row_frame, text="👁 3D", width=65, height=26, font=self.font_small,
                    fg_color=ACCENT_CYAN, hover_color=ACCENT_CYAN_HOVER,
                    command=lambda fn=fname: self.open_case_3d(fn)
                )
                btn_3d.pack(side="right", padx=8, pady=4)

            self.update_graph()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Chyba načtení", f"Nepodařilo se načíst výsledky:\n{e}")

    def open_case_3d(self, filename):
        """1-click open PyVista 3D viewer for a specific case from the table."""
        if not self.current_csv_dir:
            return

        mask_candidate = os.path.join(self.current_csv_dir, f"mask_{filename}")
        if not os.path.exists(mask_candidate):
            mask_candidate = os.path.join(self.current_csv_dir, filename)

        if os.path.exists(mask_candidate):
            self.viz_primary_var.set(mask_candidate)
            self._update_detection_info(mask_candidate)
            self.tabview.set("🚀 3D Prohlížeč (PyVista)")
            self.action_launch_pyvista()
        else:
            messagebox.showinfo("Soubor nenalezen", f"Maska pro '{filename}' nebyla nalezena ve složce:\n{self.current_csv_dir}")

    def update_graph(self, *args):
        if self.df_patient is None or self.df_patient.empty:
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            metric = self.metric_var.get()
            if metric not in self.df_patient.columns:
                return

            for widget in self.frame_graph.winfo_children():
                widget.destroy()

            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor=BG_CARD)
            ax.set_facecolor(BG_INPUT)
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(BORDER_COLOR)

            y_vals = self.df_patient[metric]
            x_vals = range(len(y_vals))

            ax.plot(
                x_vals, y_vals, marker="o", color=ACCENT_CYAN,
                linewidth=2, markersize=6, markerfacecolor=ACCENT_GREEN
            )
            ax.set_title(f"Trend: {metric}", color=TEXT_MAIN, fontsize=11, fontweight="bold")
            ax.grid(True, color=BORDER_COLOR, linestyle="--", alpha=0.4)

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.frame_graph)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════
    # TAB 4: ⚙ NASTAVENÍ (SETTINGS)
    # ═══════════════════════════════════════════════════════════════════
    def build_settings_tab(self):
        self.tab_settings.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self.tab_settings, text="Konfigurace modelů a referencí",
            font=self.font_title, text_color=TEXT_MAIN
        ).grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="w")

        self.sv_ref = ctk.StringVar(value=self.settings.get("anaknee_ref_mri", ""))
        self.sv_model = ctk.StringVar(value=self.settings.get("model_ckpt", ""))
        self.sv_ens = ctk.StringVar(value=self.settings.get("ensemble_dir", ""))
        self.sv_gt = ctk.StringVar(value=self.settings.get("gt_masks_dir", ""))

        self._add_setting_row("Referenční MRI:", self.sv_ref, 1)
        self._add_setting_row("Váhy modelu:", self.sv_model, 2)
        self._add_setting_row("Složka 5-Fold modelů:", self.sv_ens, 3, is_dir=True)
        self._add_setting_row("GT masky (validace):", self.sv_gt, 4, is_dir=True)

        ctk.CTkButton(
            self.tab_settings, text="💾  Uložit nastavení",
            font=self.font_btn, fg_color=ACCENT_GREEN, hover_color=ACCENT_GREEN_HOVER,
            height=38, corner_radius=8, command=self.save_settings_from_ui
        ).grid(row=5, column=0, columnspan=3, padx=15, pady=20, sticky="ew")

    def _add_setting_row(self, label, string_var, row, is_dir=False):
        ctk.CTkLabel(self.tab_settings, text=label, font=self.font_body).grid(row=row, column=0, padx=(15, 6), pady=8, sticky="e")
        ctk.CTkEntry(self.tab_settings, textvariable=string_var, font=self.font_small, fg_color=BG_INPUT, border_color=BORDER_COLOR).grid(row=row, column=1, padx=6, pady=8, sticky="ew")

        def browse():
            path = filedialog.askdirectory() if is_dir else filedialog.askopenfilename()
            if path:
                string_var.set(path)

        ctk.CTkButton(self.tab_settings, text="Procházet...", width=95, font=self.font_body, fg_color=ACCENT_CYAN, hover_color=ACCENT_CYAN_HOVER, command=browse).grid(row=row, column=2, padx=(6, 15), pady=8)

    def save_settings_from_ui(self):
        self.settings["anaknee_ref_mri"] = self.sv_ref.get().strip()
        self.settings["model_ckpt"] = self.sv_model.get().strip()
        self.settings["ensemble_dir"] = self.sv_ens.get().strip()
        self.settings["gt_masks_dir"] = self.sv_gt.get().strip()
        self.save_settings()
        messagebox.showinfo("Uloženo", "Nastavení bylo úspěšně uloženo.")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()

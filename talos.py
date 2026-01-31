import customtkinter as ctk
from tkinter import filedialog
import subprocess
import threading
import os
import json
import sys
import multiprocessing

# Color Palette for Cyber Look (Dicto Same)
THEME_GREEN = "#2ecc71"
THEME_RED = "#e74c3c"
THEME_ORANGE = "#f39c12"
THEME_DARK = "#1a1a1a"
CONSOLE_BG = "#000000"

ctk.set_appearance_mode("Dark")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class WAFProSoftware(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Talos - The Automaton Engine: Building ML Shields for Modern Spaces")
        self.geometry("1100x750")
        self.selected_file = None 
        self.test_file = None 
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color="#111111")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.logo = ctk.CTkLabel(self.sidebar, text="🛡️ Talos Engine", font=ctk.CTkFont(size=22, weight="bold", family="Orbitron"))
        self.logo.pack(pady=(40, 10))
        self.status_tag = ctk.CTkLabel(self.sidebar, text="ML ENGINE ACTIVE", text_color=THEME_GREEN, font=("Consolas", 10))
        self.status_tag.pack(pady=(0, 30))

        # --- Training Section ---
        self.h1 = ctk.CTkLabel(self.sidebar, text="[ MODEL PIPELINE ]", font=ctk.CTkFont(size=13, weight="bold"))
        self.h1.pack(pady=(10, 5))
        self.btn_browse = ctk.CTkButton(self.sidebar, text="📂 LOAD DATASET", fg_color="transparent", border_width=1, command=self.browse_json)
        self.btn_browse.pack(pady=10, padx=20, fill="x")
        self.btn_train = ctk.CTkButton(self.sidebar, text="⚙️ START TRAINING", fg_color=THEME_GREEN, text_color="black", state="disabled", command=self.start_training)
        self.btn_train.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="—" * 15, text_color="#444444").pack(pady=10)

        # --- Analysis Section ---
        self.h2 = ctk.CTkLabel(self.sidebar, text="[ TALOS ANALYSIS ]", font=ctk.CTkFont(size=13, weight="bold"))
        self.h2.pack(pady=(5, 5))
        self.payload_box = ctk.CTkTextbox(self.sidebar, height=120, corner_radius=5, border_width=1, fg_color="#0a0a0a", font=("Consolas", 11))
        self.payload_box.pack(pady=10, padx=20, fill="x")
        self.btn_live_test = ctk.CTkButton(self.sidebar, text="🚀 FIRE PAYLOAD", fg_color=THEME_ORANGE, text_color="black", command=self.run_live_test)
        self.btn_live_test.pack(pady=5, padx=20, fill="x")
        self.btn_browse_test = ctk.CTkButton(self.sidebar, text="📄 SCAN JSON FILE", fg_color="#3498db", command=self.browse_test_file)
        self.btn_browse_test.pack(pady=10, padx=20, fill="x")

        self.status_indicator = ctk.CTkLabel(self.sidebar, text="Made by: @Ghosthets", text_color="gray", font=("Consolas", 11))
        self.status_indicator.pack(side="bottom", pady=20)

        # --- Console ---
        self.console_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#000000")
        self.console_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.output_box = ctk.CTkTextbox(self.console_frame, fg_color=CONSOLE_BG, text_color="#00FF00", font=("Consolas", 13))
        self.output_box.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        self.prog_bar = ctk.CTkProgressBar(self.console_frame, height=8, progress_color=THEME_GREEN)
        self.prog_bar.set(0)
        self.prog_bar.pack(fill="x", padx=15, pady=(0, 20))

    def write_log(self, text):
        self.output_box.insert("end", f">>> {text}\n")
        self.output_box.see("end")

    def browse_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            self.selected_file = file_path
            self.write_log(f"SOURCE_DATASET: {os.path.basename(file_path)} LOADED SUCCESS")
            self.btn_train.configure(state="normal")

    def browse_test_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path: self.run_inference(file_path)

    def run_live_test(self):
        payload = self.payload_box.get("1.0", "end-1c").strip()
        if not payload:
            self.write_log("CRITICAL: Input Buffer Empty.")
            return
        os.makedirs("tests", exist_ok=True)
        temp_path = os.path.abspath(os.path.join("tests", "live_temp.json"))
        with open(temp_path, "w", encoding='utf-8') as f:
            json.dump({"request_data": {"query_string": f"q={payload}"}}, f)
        self.run_inference(temp_path)

    def run_command(self, cmd, task_name):
        def worker():
            self.prog_bar.set(0.2)
            try:
                # Talos By Ghosthets
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                    text=True, encoding='utf-8', bufsize=1, universal_newlines=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                for line in iter(process.stdout.readline, ""):
                    if line: self.write_log(line.strip())
                process.stdout.close()
                rc = process.wait()
                self.prog_bar.set(1 if rc == 0 else 0)
                self.write_log(f"STATUS: {task_name} COMPLETED.")
            except Exception as e:
                self.write_log(f"SYSTEM ERROR: {str(e)}")
        threading.Thread(target=worker, daemon=True).start()

    def start_training(self):
        script = resource_path(os.path.join("scripts", "train.py"))
        data = os.path.abspath(self.selected_file)
        # Made By Ghosthets
        cmd = [sys.executable, "-u", script, "--data", data]
        self.write_log("SYSTEM: Booting Intelligence Pipeline...")
        self.run_command(cmd, "TRAINING")

    def run_inference(self, file_path):
        script = resource_path(os.path.join("scripts", "inference.py"))
        target = os.path.abspath(file_path)
        cmd = [sys.executable, "-u", script, target]
        self.run_command(cmd, "ANALYSIS")

if __name__ == "__main__":
    multiprocessing.freeze_support() #@Ghosthets
    app = WAFProSoftware()
    app.mainloop()
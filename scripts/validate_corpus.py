"""
validate_corpus.py — Manual text corpus validator

Navigation:  ← / →  (or A / D)   — previous / next
Labeling:    0  →  exclude        1  →  include
Save:        Ctrl+S               — save labels CSV
Quit:        Ctrl+Q               — save and quit

Labels are written to  <original_name>_labels.csv  as  [id_col, label]
so they can be merged back later:
    pd.read_csv("data.csv").merge(pd.read_csv("data_labels.csv"), on="id")
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import pandas as pd
import sys
import os

# ── palette ──────────────────────────────────────────────────────────────────
BG        = "#1e1e2e"
SURFACE   = "#2a2a3e"
ACCENT    = "#7c6af7"
GREEN     = "#a6e3a1"
RED       = "#f38ba8"
YELLOW    = "#f9e2af"
TEXT      = "#cdd6f4"
SUBTEXT   = "#6c7086"
BORDER    = "#313244"
FONT_MONO = ("JetBrains Mono", 12) if sys.platform != "darwin" else ("Menlo", 12)
FONT_UI   = ("Segoe UI", 10)       if sys.platform != "darwin" else ("SF Pro Text", 10)

# ── state ─────────────────────────────────────────────────────────────────────
state = {
    "df": None,
    "col": None,
    "id_col": None,
    "label_col": "label",
    "path": None,
    "labels_path": None,
    "idx": 0,
}

# ── helpers ───────────────────────────────────────────────────────────────────

def labels_path_for(csv_path):
    base, _ = os.path.splitext(csv_path)
    return base + "_labels.csv"


def load_file():
    path = filedialog.askopenfilename(
        title="Open CSV dataset",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not path:
        return False

    df = pd.read_csv(path)
    if df.empty:
        messagebox.showerror("Error", "The CSV file is empty.")
        return False

    cols = list(df.columns)

    # pick text column
    if len(cols) == 1:
        col = cols[0]
        id_col = None
    else:
        col = simpledialog.askstring(
            "Select text column",
            f"Available columns:\n{', '.join(cols)}\n\nEnter the TEXT column name:",
            initialvalue=cols[0],
        )
        if col not in cols:
            messagebox.showerror("Error", f"Column '{col}' not found.")
            return False

        # pick id column
        remaining = [c for c in cols if c != col]
        id_col = simpledialog.askstring(
            "Select ID column",
            f"Available columns:\n{', '.join(remaining)}\n\nEnter the ID column name\n(leave blank to use row index):",
            initialvalue=remaining[0],
        )
        if id_col and id_col not in cols:
            messagebox.showerror("Error", f"Column '{id_col}' not found.")
            return False
        if not id_col:
            id_col = None

    lpath = labels_path_for(path)

    # initialise label column from existing labels file, if any
    df[state["label_col"]] = pd.NA
    if os.path.exists(lpath):
        try:
            existing = pd.read_csv(lpath)
            if id_col and id_col in existing.columns:
                df = df.merge(
                    existing[[id_col, state["label_col"]]],
                    on=id_col, how="left",
                    suffixes=("_orig", ""),
                )
                if state["label_col"] + "_orig" in df.columns:
                    df.drop(columns=[state["label_col"] + "_orig"], inplace=True)
            elif state["label_col"] in existing.columns:
                df[state["label_col"]] = existing[state["label_col"]].values
        except Exception:
            pass  # corrupted labels file — start fresh

    # resume from first unlabelled row
    unlabelled = df[df[state["label_col"]].isna()].index
    start_idx = int(unlabelled[0]) if len(unlabelled) else 0

    state.update({
        "df": df, "col": col, "id_col": id_col,
        "path": path, "labels_path": lpath, "idx": start_idx,
    })
    return True


def save_file(quiet=False):
    df = state["df"]
    lpath = state["labels_path"]
    id_col = state["id_col"]
    flush_edit()

    labelled = df[df[state["label_col"]].notna()]
    if id_col:
        out = labelled[[id_col, state["label_col"]]].copy()
    else:
        out = labelled[[state["label_col"]]].copy()
        out.index.name = "row_index"
        out = out.reset_index()

    out.to_csv(lpath, index=False)
    if not quiet:
        status_var.set(f"  ✓  Labels saved → {os.path.basename(lpath)}")


def flush_edit():
    """Write text box content back to the dataframe."""
    if state["df"] is None:
        return
    new_text = text_box.get("1.0", tk.END).rstrip("\n")
    state["df"].at[state["idx"], state["col"]] = new_text


def go_to(new_idx):
    flush_edit()
    n = len(state["df"])
    state["idx"] = max(0, min(new_idx, n - 1))
    render()


def assign_label(value):
    flush_edit()
    state["df"].at[state["idx"], state["label_col"]] = value
    save_file(quiet=True)
    # auto-advance if not at end
    if state["idx"] < len(state["df"]) - 1:
        go_to(state["idx"] + 1)
    else:
        render()

# ── rendering ─────────────────────────────────────────────────────────────────

def render():
    df, idx, col = state["df"], state["idx"], state["col"]
    n = len(df)
    pct = (idx + 1) / n * 100

    # progress
    progress_var.set(pct)
    pct_label.config(text=f"{idx + 1} / {n}   ({pct:.1f}%)")

    # label badge
    lbl = df.at[idx, state["label_col"]]
    if pd.isna(lbl):
        label_badge.config(text="  —  unlabelled  ", bg=SURFACE, fg=SUBTEXT)
    elif int(lbl) == 1:
        label_badge.config(text="  ✓  INCLUDE  ", bg=GREEN, fg="#1e1e2e")
    else:
        label_badge.config(text="  ✗  EXCLUDE  ", bg=RED,   fg="#1e1e2e")

    # text content
    text_box.config(state=tk.NORMAL)
    text_box.delete("1.0", tk.END)
    text_box.insert("1.0", str(df.at[idx, col]))

    # status bar
    labelled = int(df[state["label_col"]].notna().sum())
    lname = os.path.basename(state["labels_path"])
    status_var.set(f"  {labelled}/{n} labelled   |   column: {col}   |   labels → {lname}")


# ── keyboard bindings ─────────────────────────────────────────────────────────

def on_key(event):
    key = event.keysym.lower()
    # navigate
    if key in ("left", "a"):
        go_to(state["idx"] - 1)
    elif key in ("right", "d"):
        go_to(state["idx"] + 1)
    # label
    elif key == "down":
        assign_label(0)
    elif key == "up":
        assign_label(1)


def on_ctrl_s(event):
    save_file()


def on_ctrl_q(event):
    save_file(quiet=True)
    root.destroy()


# ── GUI build ──────────────────────────────────────────────────────────────────

root = tk.Tk()
root.title("Corpus Validator")
root.configure(bg=BG)
root.geometry("900x680")
root.minsize(700, 500)

# ── header row ────────────────────────────────────────────────────────────────
header = tk.Frame(root, bg=BG, pady=8, padx=16)
header.pack(fill=tk.X)

pct_label = tk.Label(header, text="0 / 0   (0.0%)", bg=BG, fg=ACCENT,
                      font=("Segoe UI", 13, "bold"))
pct_label.pack(side=tk.LEFT)

label_badge = tk.Label(header, text="  —  unlabelled  ", bg=SURFACE, fg=SUBTEXT,
                        font=("Segoe UI", 11, "bold"), relief="flat", padx=6, pady=4)
label_badge.pack(side=tk.RIGHT, padx=(0, 4))

# ── progress bar ──────────────────────────────────────────────────────────────
pb_frame = tk.Frame(root, bg=BG, padx=16)
pb_frame.pack(fill=tk.X)

style = ttk.Style()
style.theme_use("clam")
style.configure("Custom.Horizontal.TProgressbar",
                 troughcolor=SURFACE, background=ACCENT,
                 bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT, thickness=8)

progress_var = tk.DoubleVar()
pb = ttk.Progressbar(pb_frame, variable=progress_var, maximum=100,
                      style="Custom.Horizontal.TProgressbar")
pb.pack(fill=tk.X, pady=(0, 10))

# ── text area ─────────────────────────────────────────────────────────────────
text_frame = tk.Frame(root, bg=BORDER, padx=1, pady=1)
text_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 6))

text_inner = tk.Frame(text_frame, bg=SURFACE)
text_inner.pack(fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(text_inner, bg=SURFACE, troughcolor=SURFACE,
                          activebackground=ACCENT, relief="flat")
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

text_box = tk.Text(
    text_inner,
    wrap=tk.WORD,
    font=FONT_MONO,
    bg=SURFACE, fg=TEXT,
    insertbackground=ACCENT,
    selectbackground=ACCENT, selectforeground="#1e1e2e",
    relief="flat",
    padx=14, pady=12,
    yscrollcommand=scrollbar.set,
    undo=True,
)
text_box.pack(fill=tk.BOTH, expand=True)
scrollbar.config(command=text_box.yview)

# ── shortcut legend ───────────────────────────────────────────────────────────
legend_frame = tk.Frame(root, bg=BG, padx=16)
legend_frame.pack(fill=tk.X, pady=(0, 4))

shortcuts = [
    ("← / A", "prev"),
    ("→ / D", "next"),
    ("0", "exclude"),
    ("1", "include"),
    ("Ctrl+S", "save"),
    ("Ctrl+Q", "quit"),
]

for i, (key, action) in enumerate(shortcuts):
    tk.Label(legend_frame, text=key, bg=BORDER, fg=YELLOW,
             font=("Segoe UI", 9, "bold"), padx=6, pady=2, relief="flat"
             ).grid(row=0, column=i*2, padx=(0 if i else 0, 2), pady=2)
    tk.Label(legend_frame, text=action, bg=BG, fg=SUBTEXT,
             font=("Segoe UI", 9)
             ).grid(row=0, column=i*2+1, padx=(0, 14), pady=2)

# ── status bar ────────────────────────────────────────────────────────────────
status_var = tk.StringVar(value="  Open a CSV to get started")
status_bar = tk.Label(root, textvariable=status_var, bg=SURFACE, fg=SUBTEXT,
                       font=("Segoe UI", 9), anchor="w", pady=4)
status_bar.pack(fill=tk.X, side=tk.BOTTOM)

# ── bindings ──────────────────────────────────────────────────────────────────
root.bind("<Key>", on_key)
root.bind("<Control-s>", on_ctrl_s)
root.bind("<Control-q>", on_ctrl_q)

# ── startup ───────────────────────────────────────────────────────────────────
root.after(100, lambda: (load_file() and render()))

root.mainloop()

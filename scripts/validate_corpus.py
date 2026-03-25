"""
validate_corpus.py — Manual text corpus validator

Navigation:   ← / →  (or A / D)  — previous / next
Labeling:     0  →  exclude       1  →  include
Update text:  click "Update text" — flush edits to original CSV
Save labels:  Ctrl+S              — write _labels.csv
Quit:         Ctrl+Q              — save labels and quit
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import pandas as pd
import sys, os

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#1e1e2e"
SURFACE = "#2a2a3e"
ACCENT  = "#7c6af7"
GREEN   = "#a6e3a1"
RED     = "#f38ba8"
YELLOW  = "#f9e2af"
ORANGE  = "#fab387"
TEXT    = "#cdd6f4"
SUBTEXT = "#6c7086"
BORDER  = "#313244"
FONT    = ("JetBrains Mono", 11) if sys.platform != "darwin" else ("Menlo", 11)

# ── state ─────────────────────────────────────────────────────────────────────
state = {
    "df"            : None,
    "col"           : None,       # text column
    "id_col"        : None,
    "right_cols"    : [],         # columns shown in right panel (<=2)
    "original_cols" : [],         # columns in the original CSV (for text save)
    "label_col"     : "label",
    "path"          : None,       # original CSV path
    "labels_path"   : None,       # _labels.csv path
    "idx"           : 0,
    "keywords"      : [],
    "text_dirty"    : False,
}

# ── file helpers ──────────────────────────────────────────────────────────────

def labels_path_for(csv_path):
    base, _ = os.path.splitext(csv_path)
    return base + "_labels.csv"


def load_file():
    # main CSV
    path = filedialog.askopenfilename(
        title="Open main CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not path:
        return False

    df = pd.read_csv(path)
    if df.empty:
        messagebox.showerror("Error", "The CSV is empty.")
        return False

    cols          = list(df.columns)
    original_cols = cols[:]
    df[state["label_col"]] = pd.NA

    # text column
    col = simpledialog.askstring(
        "Text column",
        f"Columns: {', '.join(cols)}\n\nText column name:",
        initialvalue=cols[0],
    )
    if col not in cols:
        messagebox.showerror("Error", f"Column '{col}' not found.")
        return False

    # id column
    id_col  = None
    remaining = [c for c in cols if c != col]
    if remaining:
        ans = simpledialog.askstring(
            "ID column",
            f"Columns: {', '.join(remaining)}\n\nID column (blank = row index):",
            initialvalue=remaining[0],
        )
        if ans and ans in cols:
            id_col = ans

    # optional: metadata CSV (title, publish_date, media_name)
    if id_col and messagebox.askyesno("Metadata", "Load a separate metadata CSV?"):
        mpath = filedialog.askopenfilename(
            title="Open metadata CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if mpath:
            mdf = pd.read_csv(mpath)
            if id_col in mdf.columns:
                merge_meta = [id_col] + [
                    c for c in ("title", "publish_date", "media_name")
                    if c in mdf.columns
                ]
                df = df.merge(mdf[merge_meta], on=id_col, how="left")
            else:
                messagebox.showwarning(
                    "Metadata merge",
                    f"ID column '{id_col}' not found in metadata CSV — skipped."
                )

    # optional: LLM predictions CSV
    right_cols = []
    if messagebox.askyesno("LLM predictions", "Load a pre-labels CSV with LLM predictions?"):
        lpath_in = filedialog.askopenfilename(
            title="Open pre-labels CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if lpath_in:
            ldf   = pd.read_csv(lpath_in)
            lcols = list(ldf.columns)

            # which column is the 0/1 pre-label?
            llm_lbl = simpledialog.askstring(
                "Pre-label column",
                f"Columns: {', '.join(lcols)}\n\nWhich column has the 0/1 class?",
                initialvalue=lcols[1] if len(lcols) > 1 else lcols[0],
            )

            # which columns go in the right panel?
            candidates = [c for c in lcols if c not in (llm_lbl, id_col)]
            if candidates:
                ans = simpledialog.askstring(
                    "Right panel columns",
                    f"Available: {', '.join(candidates)}\n\n"
                    "Enter up to 2 column names (comma-separated):",
                    initialvalue=", ".join(candidates[:2]),
                )
                if ans:
                    right_cols = [
                        c.strip() for c in ans.split(",")
                        if c.strip() in candidates
                    ][:2]

            # columns to merge
            merge_cols = []
            if id_col and id_col in lcols:
                merge_cols.append(id_col)
            if llm_lbl and llm_lbl in lcols:
                merge_cols.append(llm_lbl)
            merge_cols += [c for c in right_cols if c in lcols]

            if id_col and id_col in lcols:
                df = df.merge(ldf[merge_cols], on=id_col, how="left")
            else:
                for c in merge_cols:
                    if c in ldf.columns:
                        df[c] = ldf[c].values

            # seed label column from LLM prediction
            if llm_lbl and llm_lbl in df.columns:
                df[state["label_col"]] = df[llm_lbl]

    # resume from existing _labels.csv
    lpath = labels_path_for(path)
    if os.path.exists(lpath):
        try:
            existing = pd.read_csv(lpath)
            if (id_col and id_col in existing.columns
                    and state["label_col"] in existing.columns):
                existing = existing.rename(
                    columns={state["label_col"]: "_lbl_ex"}
                )
                df = df.merge(existing[[id_col, "_lbl_ex"]], on=id_col, how="left")
                mask = df["_lbl_ex"].notna()
                df.loc[mask, state["label_col"]] = df.loc[mask, "_lbl_ex"]
                df.drop(columns=["_lbl_ex"], inplace=True)
            elif state["label_col"] in existing.columns:
                mask = existing[state["label_col"]].notna()
                df.loc[mask.values, state["label_col"]] = (
                    existing.loc[mask, state["label_col"]].values
                )
        except Exception:
            pass

    unlabelled = df[df[state["label_col"]].isna()].index
    start_idx  = int(unlabelled[0]) if len(unlabelled) else 0

    # normalize label column to float to avoid pd.NA / dtype issues
    df[state["label_col"]] = pd.to_numeric(df[state["label_col"]], errors="coerce")

    # debug: print label distribution to console
    lcol = state["label_col"]
    print(f"[load] label col dtype: {df[lcol].dtype}")
    print(f"[load] value_counts:\n{df[lcol].value_counts(dropna=False)}")

    state.update({
        "df"            : df,
        "col"           : col,
        "id_col"        : id_col,
        "right_cols"    : right_cols,
        "original_cols" : original_cols,
        "path"          : path,
        "labels_path"   : lpath,
        "idx"           : start_idx,
    })
    return True


def save_labels(quiet=False):
    df     = state["df"]
    lpath  = state["labels_path"]
    id_col = state["id_col"]

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


def save_text_edit():
    """Flush text box content to df and overwrite original CSV (original cols only)."""
    if state["df"] is None:
        return
    new_text = text_box.get("1.0", tk.END).rstrip("\n")
    state["df"].at[state["idx"], state["col"]] = new_text

    orig = state["original_cols"]
    state["df"][[c for c in orig if c in state["df"].columns]].to_csv(
        state["path"], index=False
    )
    mark_clean()
    show_toast("✎  Text updated", ACCENT)
    status_var.set(f"  ✓  Text updated → {os.path.basename(state['path'])}")


def flush_text_silently():
    """Auto-flush text on navigation without writing to disk."""
    if state["text_dirty"] and state["df"] is not None:
        new_text = text_box.get("1.0", tk.END).rstrip("\n")
        state["df"].at[state["idx"], state["col"]] = new_text
        mark_clean()

# ── dirty / clean ─────────────────────────────────────────────────────────────

def mark_clean():
    state["text_dirty"] = False
    update_btn.config(bg=SURFACE, fg=SUBTEXT)


def mark_dirty():
    if not state["text_dirty"]:
        state["text_dirty"] = True
        update_btn.config(bg=ORANGE, fg="#1e1e2e")

# ── toast notification ────────────────────────────────────────────────────────

_toast_job = None

def show_toast(text, color=ACCENT):
    global _toast_job
    toast_label.config(text=f"  {text}  ", bg=color, fg="#1e1e2e")
    toast_label.place(relx=0.5, rely=0.0, anchor="n", y=6)
    if _toast_job:
        root.after_cancel(_toast_job)
    _toast_job = root.after(1800, lambda: toast_label.place_forget())

# ── keyword highlighting ───────────────────────────────────────────────────────

def apply_highlights():
    text_box.tag_remove("highlight", "1.0", tk.END)
    for kw in state["keywords"]:
        if not kw:
            continue
        start = "1.0"
        while True:
            pos = text_box.search(kw, start, tk.END, nocase=True, regexp=False)
            if not pos:
                break
            end_pos = f"{pos}+{len(kw)}c"
            text_box.tag_add("highlight", pos, end_pos)
            start = end_pos


def on_keywords_change(event=None):
    raw = keywords_entry.get()
    state["keywords"] = [k.strip() for k in raw.split(",") if k.strip()]
    apply_highlights()

# ── navigation / labeling ─────────────────────────────────────────────────────

def go_to(new_idx):
    flush_text_silently()
    n = len(state["df"])
    state["idx"] = max(0, min(new_idx, n - 1))
    render()


def assign_label(value):
    flush_text_silently()
    prev = state["df"].at[state["idx"], state["label_col"]]
    state["df"].at[state["idx"], state["label_col"]] = value
    save_labels(quiet=True)

    if pd.isna(prev):
        prev_str = "—"
    else:
        prev_str = "INCLUDE" if int(prev) == 1 else "EXCLUDE"
    new_str  = "INCLUDE" if value == 1 else "EXCLUDE"
    color    = GREEN if value == 1 else RED
    if str(prev) != str(value):
        show_toast(f"{prev_str}  →  {new_str}", color)
    else:
        show_toast(f"✓  {new_str}", color)

    if state["idx"] < len(state["df"]) - 1:
        go_to(state["idx"] + 1)
    else:
        render()

# ── rendering ─────────────────────────────────────────────────────────────────

def render():
    df, idx, col = state["df"], state["idx"], state["col"]
    n   = len(df)
    pct = (idx + 1) / n * 100

    progress_var.set(pct)
    pct_label.config(text=f"{idx + 1} / {n}   ({pct:.1f}%)")

    # stats breakdown — cast to numeric to handle float/object dtypes after merges
    lcol   = state["label_col"]
    labels = pd.to_numeric(df[lcol], errors="coerce")
    n1     = int((labels == 1).sum())
    n0     = int((labels == 0).sum())
    nna    = int(labels.isna().sum())
    stats_label.config(
        text=f"  ✓ {n1} ({n1/n*100:.0f}%)   ✗ {n0} ({n0/n*100:.0f}%)   — {nna} ({nna/n*100:.0f}%)"
    )

    lbl = pd.to_numeric(df.at[idx, state["label_col"]], errors="coerce")
    if pd.isna(lbl):
        label_badge.config(text="  —  unlabelled  ", bg=SURFACE, fg=SUBTEXT)
    elif int(lbl) == 1:
        label_badge.config(text="  ✓  INCLUDE  ", bg=GREEN, fg="#1e1e2e")
    else:
        label_badge.config(text="  ✗  EXCLUDE  ", bg=RED,   fg="#1e1e2e")

    # metadata strip
    for col_name, var in meta_vars.items():
        val = df.at[idx, col_name] if col_name in df.columns else ""
        var.set("" if pd.isna(val) else str(val))

    # left panel: editable text
    text_box.config(state=tk.NORMAL)
    text_box.delete("1.0", tk.END)
    text_box.insert("1.0", str(df.at[idx, col]))
    text_box.edit_modified(False)   # reset so <<Modified>> won't fire yet
    apply_highlights()
    mark_clean()

    # right panels: read-only
    for i, rc in enumerate(state["right_cols"][:2]):
        box = right_boxes[i]
        box.config(state=tk.NORMAL)
        box.delete("1.0", tk.END)
        val = df.at[idx, rc] if rc in df.columns else ""
        box.insert("1.0", "" if pd.isna(val) else str(val))
        box.config(state=tk.DISABLED)

    root.focus_set()

    labelled = int(df[state["label_col"]].notna().sum())
    lname    = os.path.basename(state["labels_path"])
    id_hint  = f"  |  id: {state['id_col']}" if state["id_col"] else ""
    status_var.set(
        f"  {labelled}/{n} labelled   |   col: {col}{id_hint}   |   → {lname}"
    )

# ── keyboard bindings ─────────────────────────────────────────────────────────

def on_key(event):
    key = event.keysym.lower()
    if key in ("left", "a"):
        go_to(state["idx"] - 1)
    elif key in ("right", "d"):
        go_to(state["idx"] + 1)
    elif key == "0":
        assign_label(0)
    elif key == "1":
        assign_label(1)


def on_ctrl_s(event):  save_labels()
def on_ctrl_q(event):  save_labels(quiet=True); root.destroy()

# ── GUI ───────────────────────────────────────────────────────────────────────

root = tk.Tk()
root.title("Corpus Validator")
root.configure(bg=BG)
root.geometry("1280x720")
root.minsize(900, 500)

# header
header = tk.Frame(root, bg=BG, pady=8, padx=16)
header.pack(fill=tk.X)

pct_label = tk.Label(header, text="0 / 0   (0.0%)", bg=BG, fg=ACCENT,
                      font=("Segoe UI", 13, "bold"))
pct_label.pack(side=tk.LEFT)

stats_label = tk.Label(header, text="", bg=BG, fg=SUBTEXT,
                        font=("Segoe UI", 9), padx=12)
stats_label.pack(side=tk.LEFT, pady=(2, 0))

label_badge = tk.Label(header, text="  —  unlabelled  ", bg=SURFACE, fg=SUBTEXT,
                        font=("Segoe UI", 11, "bold"), padx=6, pady=4)
label_badge.pack(side=tk.RIGHT, padx=(0, 4))

# progress bar
pb_frame = tk.Frame(root, bg=BG, padx=16)
pb_frame.pack(fill=tk.X)

style = ttk.Style()
style.theme_use("clam")
style.configure("V.Horizontal.TProgressbar",
                 troughcolor=SURFACE, background=ACCENT,
                 bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT, thickness=8)
progress_var = tk.DoubleVar()
pb = ttk.Progressbar(pb_frame, variable=progress_var, maximum=100,
                      style="V.Horizontal.TProgressbar")
pb.pack(fill=tk.X, pady=(0, 6))

# toast (hidden until triggered, floats over content)
toast_label = tk.Label(root, text="", font=("Segoe UI", 10, "bold"),
                        padx=14, pady=6, relief="flat")
# (placed/hidden dynamically by show_toast)

# keywords bar
kw_frame = tk.Frame(root, bg=BG, padx=16, pady=3)
kw_frame.pack(fill=tk.X)
tk.Label(kw_frame, text="Highlight terms:", bg=BG, fg=SUBTEXT,
          font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 8))
keywords_entry = tk.Entry(kw_frame, bg=SURFACE, fg=YELLOW, insertbackground=ACCENT,
                           relief="flat", font=("Segoe UI", 9), bd=4)
keywords_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
keywords_entry.bind("<KeyRelease>", on_keywords_change)
tk.Label(kw_frame, text="comma-separated", bg=BG, fg=SUBTEXT,
          font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(8, 0))

# ── split pane ────────────────────────────────────────────────────────────────
paned = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg=BORDER,
                        sashwidth=5, sashrelief="flat", bd=0)
paned.pack(fill=tk.BOTH, expand=True, padx=16, pady=6)

# LEFT: editable text
left_frame = tk.Frame(paned, bg=BG)
paned.add(left_frame, minsize=350, stretch="always")

meta_frame = tk.Frame(left_frame, bg=BG)
meta_frame.pack(fill=tk.X, pady=(0, 6))

meta_fields = [
    ("title",        "Title"),
    ("publish_date", "Date"),
    ("media_name",   "Media"),
]
meta_vars = {}

for col_name, display in meta_fields:
    row = tk.Frame(meta_frame, bg=BG)
    row.pack(fill=tk.X, pady=1)
    tk.Label(row, text=display, bg=BG, fg=SUBTEXT,
              font=("Segoe UI", 8, "bold"), width=6, anchor="w"
              ).pack(side=tk.LEFT, padx=(2, 6))
    var = tk.StringVar()
    meta_vars[col_name] = var
    tk.Entry(row, textvariable=var, state="readonly",
             readonlybackground=SURFACE, fg=TEXT,
             relief="flat", font=("Segoe UI", 9), bd=4,
             ).pack(side=tk.LEFT, fill=tk.X, expand=True)

tk.Label(left_frame, text="TEXT", bg=BG, fg=SUBTEXT,
          font=("Segoe UI", 8, "bold"), anchor="w").pack(fill=tk.X, padx=2, pady=(0, 2))

text_outer = tk.Frame(left_frame, bg=BORDER, padx=1, pady=1)
text_outer.pack(fill=tk.BOTH, expand=True)
text_inner_f = tk.Frame(text_outer, bg=SURFACE)
text_inner_f.pack(fill=tk.BOTH, expand=True)

text_scroll = tk.Scrollbar(text_inner_f, bg=SURFACE, troughcolor=SURFACE,
                             activebackground=ACCENT, relief="flat")
text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

text_box = tk.Text(
    text_inner_f, wrap=tk.WORD, font=FONT, bg=SURFACE, fg=TEXT,
    insertbackground=ACCENT, selectbackground=ACCENT, selectforeground="#1e1e2e",
    relief="flat", padx=14, pady=12, yscrollcommand=text_scroll.set, undo=True,
)
text_box.pack(fill=tk.BOTH, expand=True)
text_scroll.config(command=text_box.yview)
text_box.tag_configure("highlight", background=YELLOW, foreground="#1e1e2e")
# fires only when user edits (reset after each trigger so it fires once per edit session)
text_box.bind("<<Modified>>", lambda e: (text_box.edit_modified(False), mark_dirty()))

update_btn = tk.Button(
    left_frame, text="Update text", bg=SURFACE, fg=SUBTEXT,
    font=("Segoe UI", 9), relief="flat", padx=10, pady=6,
    cursor="hand2", activebackground=ACCENT, activeforeground="#1e1e2e",
    command=save_text_edit,
)
update_btn.pack(fill=tk.X, pady=(4, 0))

# RIGHT: two read-only panels
right_frame = tk.Frame(paned, bg=BG)
paned.add(right_frame, minsize=220, stretch="always")

right_boxes          = []
right_label_widgets  = []

for i in range(2):
    panel = tk.Frame(right_frame, bg=BG)
    panel.pack(fill=tk.BOTH, expand=True, pady=(0, 6 if i == 0 else 0))

    rlbl = tk.Label(panel, text="—", bg=BG, fg=SUBTEXT,
                     font=("Segoe UI", 8, "bold"), anchor="w")
    rlbl.pack(fill=tk.X, padx=2, pady=(0, 2))
    right_label_widgets.append(rlbl)

    router = tk.Frame(panel, bg=BORDER, padx=1, pady=1)
    router.pack(fill=tk.BOTH, expand=True)
    rinner = tk.Frame(router, bg=SURFACE)
    rinner.pack(fill=tk.BOTH, expand=True)

    rscroll = tk.Scrollbar(rinner, bg=SURFACE, troughcolor=SURFACE,
                            activebackground=ACCENT, relief="flat")
    rscroll.pack(side=tk.RIGHT, fill=tk.Y)

    rbox = tk.Text(
        rinner, wrap=tk.WORD, font=FONT, bg=SURFACE, fg=TEXT,
        relief="flat", padx=10, pady=8,
        yscrollcommand=rscroll.set, state=tk.DISABLED, cursor="arrow",
    )
    rbox.pack(fill=tk.BOTH, expand=True)
    rscroll.config(command=rbox.yview)
    right_boxes.append(rbox)

# legend
legend_frame = tk.Frame(root, bg=BG, padx=16)
legend_frame.pack(fill=tk.X, pady=(0, 2))

shortcuts = [
    ("← / A", "prev"), ("→ / D", "next"),
    ("0", "exclude"),   ("1", "include"),
    ("Ctrl+S", "save labels"), ("Ctrl+Q", "quit"),
]
for i, (key, action) in enumerate(shortcuts):
    tk.Label(legend_frame, text=key, bg=BORDER, fg=YELLOW,
              font=("Segoe UI", 9, "bold"), padx=6, pady=2
              ).grid(row=0, column=i * 2,     padx=(0, 2), pady=2)
    tk.Label(legend_frame, text=action, bg=BG, fg=SUBTEXT,
              font=("Segoe UI", 9)
              ).grid(row=0, column=i * 2 + 1, padx=(0, 14), pady=2)

# status bar
status_var = tk.StringVar(value="  Open a CSV to get started")
tk.Label(root, textvariable=status_var, bg=SURFACE, fg=SUBTEXT,
          font=("Segoe UI", 9), anchor="w", pady=4
          ).pack(fill=tk.X, side=tk.BOTTOM)

# bindings
root.bind("<Key>", on_key)
root.bind("<Control-s>", on_ctrl_s)
root.bind("<Control-q>", on_ctrl_q)


def startup():
    if load_file():
        for i, rc in enumerate(state["right_cols"][:2]):
            right_label_widgets[i].config(text=rc.upper())
        render()


root.after(100, startup)
root.mainloop()
import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm

class BaseWorker:
    """
    Para trabajos distribuidos en Google Colab.
    """
    def __init__(
        self,
        input_csv,
        output_csv,
        id_col,
        batch_size = 25,
        resume = True,
        total_workers = 1,
        wid = 0):

        self.id_col = id_col
        self.batch_size = batch_size
        self.resume = resume
        self._buffer = []

        df = pd.read_csv(input_csv)

        if total_workers > 1:
            output_csv = self._worker_path(output_csv, wid)
            cuts =  np.linspace(0, len(df), total_workers + 1, dtype = int)
            df = df.iloc[
                cuts[wid]:cuts[wid + 1]
            ]

        self.output_csv = output_csv
        self.df = df

        self.done = set()
        if resume and os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
            try:
                self.done = set(
                    pd.read_csv(output_csv, usecols=[id_col])[id_col].astype(str)
                )
            except Exception:
                pass
    
    @staticmethod
    def _worker_path(
        path,
        wid):

        base, ext = os.path.splitext(path)
        return f"{base}.w{wid:03d}{ext or '.csv'}"
    
    def process_row(self, row) -> dict | None:
        """Implementar en subclase."""
        raise NotImplementedError

    def on_error(self, row, exc: Exception) -> dict | None:
        """
        Comportamiento por defecto: skip silencioso.
        Sobreescribir para loggear o guardar fila de error.
        """
        print(f"✗ {row['doc_id']}: {exc}")
        return None

    def run(self):
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            if self.resume and str(row[self.id_col]) in self.done:
                continue
            try:
                result = self.process_row(row)
            except Exception as e:
                result = self.on_error(row, e)

            if result is not None:
                self._buffer.append(result)
            if len(self._buffer) >= self.batch_size:
                self._flush()
        self._flush()

    def _flush(self):
        if not self._buffer:
            return
        pd.DataFrame(self._buffer).to_csv(
            self.output_csv,
            index=False,
            mode="a",
            header=not os.path.exists(self.output_csv) or os.path.getsize(self.output_csv) == 0,
            encoding="utf-8",
        )
        self._buffer.clear()

def merge_workers(output_csv, id_col, dirwids: str = "."):
    pattern = os.path.join(
        dirwids, 
        os.path.splitext(os.path.basename(output_csv))[0] + ".w*.csv"
    )
    dfs = pd.concat(
        [pd.read_csv(f) for f in glob.glob(pattern)], 
        ignore_index=True
    )
    dfs.drop_duplicates(subset=[id_col], keep="last").to_csv(output_csv, index=False)
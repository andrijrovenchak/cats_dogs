from __future__ import annotations

import pandas as pd

from fetch_ncbi_mito_results import download_species_fastas


if __name__ == "__main__":
    df = pd.read_excel("felidae.xlsx", sheet_name="Felidae mtDNA links")
    for sp in df["Scientific name"].dropna():
        paths = download_species_fastas(sp, mode="complete", outdir="felidae_fastas")
        print(sp, "->", paths)

    df = pd.read_excel("canidae.xlsx", sheet_name="Canidae mtDNA links")
    for sp in df["Scientific name"].dropna():
        paths = download_species_fastas(sp, mode="complete", outdir="canidae_fastas")
        print(sp, "->", paths)

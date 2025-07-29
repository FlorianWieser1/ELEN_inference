"""
compute_SaProt_embeddings.py

Author: Florian Wieser
Date: 2025-07-29

Description:
    This script computes SaProt sequence embeddings for a set of PDB files.
    It allows you to specify the SaProt repository path as an argument, and will work with
    any valid installation. Results are written as .h5 (and optionally .json) files for downstream use.

Usage (example):
    python compute_SaProt_embeddings.py \
        --inpath_models natives \
        --outpath SaProt_embeddings \
        --path_saprot ~/software/SaProt/ \
        --overwrite

Required:
    - Python environment with all SaProt dependencies installed and working.
    - SaProt repository path (`--path_saprot`) must be provided.

Arguments:
    --inpath_models:   Directory containing input PDB files.
    --outpath:         Directory to store output embeddings.
    --saprot_model:    Name for output embedding file (default: saprot_650M).
    --path_saprot:     Path to local SaProt repo installation (must be provided).
    --overwrite:       Overwrite the output directory if it exists.
    --path_discarded:  Directory for problematic/discarded PDB files.
    --write_json:      Also write embeddings as a JSON file.

"""

import os
import sys
import h5py
import glob
import torch
import shutil
import argparse as ap
import json

from Bio.PDB import PDBParser

def discard_pdb(path_pdb, path_discarded, step, error, log_file="discarded_pdb.log"):
    """
    Move a problematic PDB file to the discarded folder and log the error.
    """
    os.makedirs(path_discarded, exist_ok=True)
    shutil.move(path_pdb, path_discarded)
    with open(log_file, "a") as log:
        log.write(f"{path_pdb}, Step: {step}, Error: {error}\n")

def get_residue_ids(path_pdb: str) -> list:
    """
    Parse residue identifiers from a PDB file (chain and residue number).
    """
    parser = PDBParser(QUIET=True)
    resnum_list = []
    structure = parser.get_structure("structure", path_pdb)
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':
                    chain_and_id = f"{chain.id}_{residue.id[1]}"
                    resnum_list.append(chain_and_id)
    return resnum_list

def get_saprot_sequence_embeddings(path_saprot, path_pdb):
    """
    Dynamically import SaProt modules and compute sequence embeddings for a single PDB file.
    """
    # Add SaProt repo to path, import locally so main script can run even if not installed
    sys.path.append(path_saprot)
    from utils.foldseek_util import get_struc_seq
    from model.esm.base import EsmBaseModel
    from transformers import EsmTokenizer
    import transformers
    transformers.logging.set_verbosity_error()

    # Path to foldseek binary (check existence)
    foldseek_path = os.path.join(path_saprot, "bin", "foldseek")
    if not os.path.isfile(foldseek_path):
        raise FileNotFoundError(f"Foldseek binary not found: {foldseek_path}")

    parsed_seqs = get_struc_seq(foldseek_path, path_pdb)
    if parsed_seqs:
        combined_seq = next(iter(parsed_seqs.values()))[2]
    else:
        print("No sequences found in parsed_seqs.")
        return None

    config = {
        "task": "base",
        "config_path": os.path.join(path_saprot, "weights", "PLMs", "SaProt_650M_PDB"),
        "load_pretrained": True
    }
    # Load SaProt model and tokenizer
    model = EsmBaseModel(**config)
    tokenizer = EsmTokenizer.from_pretrained(config["config_path"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = tokenizer(combined_seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embeddings = model.get_hidden_states(inputs)
    return embeddings[0]  # shape: [seq_len, emb_dim]

def calculate_sequence_embeddings(
    inpath, saprot_model, outpath, write_json=False, path_discarded="discarded", path_saprot=None
):
    """
    Main computation: iterate over PDBs, compute and store SaProt embeddings.
    """
    path_out = os.path.join(outpath, f"{saprot_model}.h5")
    path_out_json = os.path.join(outpath, f"{saprot_model}.json")
    json_embeddings = {}

    pdb_files = glob.glob(os.path.join(inpath, "*.pdb"))
    if not pdb_files:
        print(f"No .pdb files found in {inpath}")
        return

    with h5py.File(path_out, 'w') as f:
        for path_pdb in pdb_files:
            fname_pdb = os.path.basename(path_pdb)
            print(f"Computing SaProt embedding for {fname_pdb} ...")
            try:
                sequence_embedding = get_saprot_sequence_embeddings(path_saprot, path_pdb)
                res_ids = get_residue_ids(path_pdb)
                if sequence_embedding is not None:
                    dset = f.create_dataset(
                        fname_pdb,
                        data=sequence_embedding.cpu(),
                        compression="gzip"
                    )
                    dset.attrs["res_ids"] = [rid.encode("utf-8") for rid in res_ids]
                    if write_json:
                        json_embeddings[fname_pdb] = {
                            "res_ids": res_ids,
                            "embedding": sequence_embedding.cpu().tolist()
                        }
            except Exception as e:
                print(f"Error processing {fname_pdb}: {e}")
                discard_pdb(path_pdb, path_discarded, "SaProt embedding calculation", e)
                continue

    if write_json and json_embeddings:
        with open(path_out_json, "w") as f_json:
            json.dump(json_embeddings, f_json)
    print(f"\nDone. Embeddings written to {path_out}{' and ' + path_out_json if write_json else ''}")

def run_compute_saprot_embeddings(
    inpath_models, saprot_model, outpath,
    overwrite=False, write_json=False, path_discarded="discarded", path_saprot=None
):
    """
    Entry point for CLI.
    """
    if path_saprot is None or not os.path.isdir(path_saprot):
        raise ValueError("Please provide a valid path to your SaProt installation with --path_saprot")
    if not os.path.isdir(inpath_models):
        raise ValueError(f"Input directory '{inpath_models}' does not exist or is not a directory.")
    if overwrite and os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.makedirs(outpath, exist_ok=True)
    calculate_sequence_embeddings(
        inpath_models, saprot_model, outpath, write_json=write_json, path_discarded=path_discarded, path_saprot=path_saprot
    )

###############################################################################
if __name__ == "__main__":
    parser = ap.ArgumentParser(
        description="Compute SaProt sequence embeddings for all PDB files in a directory."
    )
    parser.add_argument('--inpath_models', type=str, default="AF3_models",
                        help="Directory containing input .pdb files [default: AF3_models]")
    parser.add_argument('--outpath', type=str, default="SaProt_embeddings",
                        help="Directory to store SaProt embeddings [default: SaProt_embeddings]")
    parser.add_argument('--saprot_model', type=str, default="saprot_650M",
                        help="Output embedding set name [default: saprot_650M]")
    parser.add_argument('--path_saprot', type=str, required=True,
                        help="Path to your SaProt repository/installation (required)")
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing output directory')
    parser.add_argument('--path_discarded', type=str, default="discarded",
                        help="Output directory for problematic/failed PDB files [default: discarded]")
    parser.add_argument('--write_json', action='store_true', default=False,
                        help='Also write embeddings as a JSON file')
    args = parser.parse_args()

    run_compute_saprot_embeddings(
        args.inpath_models,
        args.saprot_model,
        args.outpath,
        overwrite=args.overwrite,
        write_json=args.write_json,
        path_discarded=args.path_discarded,
        path_saprot=args.path_saprot
    )

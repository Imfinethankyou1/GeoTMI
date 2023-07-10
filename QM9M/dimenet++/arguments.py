import argparse
from pathlib import Path


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Input")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for mini-batch training."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Epochs")
    parser.add_argument(
        "--task", type=str, default="dft", choices=["dft", "mmff", "ours"]
    )
    parser.add_argument("--save_dir", type=Path, default="test", help="SAVE DIRECTORY")
    parser.add_argument(
        "--label_dir",
        type=Path,
        default="/home/share/DATA/khs_ICML/qm9_data/CV_data",
        help="LABEL DIRECTORY",
    )
    parser.add_argument("--ncpu", type=int, default=4, help="num_workers")
    parser.add_argument(
        "--hidden_channels", type=int, default=128, help="hidden channels"
    )
    parser.add_argument(
        "--n_outputs", type=int, default=1, help="Num of dimension of output"
    )
    parser.add_argument("--num_blocks", type=int, default=4, help="num_blocks")
    parser.add_argument(
        "--int_emb_size", type=int, default=64, help="interaction embedding size"
    )
    parser.add_argument(
        "--basis_emb_size", type=int, default=8, help="basis embedding size"
    )
    parser.add_argument(
        "--out_emb_channels", type=int, default=256, help="output embedding channels"
    )
    parser.add_argument("--num_spherical", type=int, default=7, help="num spherical")
    parser.add_argument("--num_radial", type=int, default=6, help="num radial")
    parser.add_argument("--cutoff", type=int, default=5.0, help="cutoff")
    parser.add_argument(
        "--envelope_exponent", type=int, default=5, help="envelope exponent"
    )
    parser.add_argument(
        "--num_before_skip", type=int, default=1, help="num before skip"
    )
    parser.add_argument("--num_after_skip", type=int, default=2, help="num after skip")
    parser.add_argument(
        "--num_output_layers", type=int, default=3, help="num output layers"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="graph data directory name",
    )
    parser.add_argument(
        "--target", type=str, default="u0_atom", help="target property name"
    )
    parser.add_argument("--dist_loss_ratio", type=float, help="dist loss ratio", default=0.1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--save_dist_results", action="store_true")
    parser.add_argument("--ngpus", type=int, default=2, help="gpus")
    parser.add_argument("--num_pos_steps", type=int, default=0, help="num pos steps")
    parser.add_argument("--checkpoint_file", type=Path, help="checkpoint file for test")
    args = parser.parse_args()
    return args

"""Convert a retargeted G1 motion (.pkl from openhe/g1-retargeted-motions) to a small .npz.

  python tools/convert_motion.py <in.pkl> <out.npz> [--start 42 --end 155]

The .npz holds fps, dof (T, 23), root_pos (T, 3) and root_quat (T, 4, wxyz) so the course
code only needs numpy to read it.
"""
import argparse
import pickle

import numpy as np


def load_pkl(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except pickle.UnpicklingError:  # some subsets (lafan1, kungfu) are saved with joblib
        import joblib
        data = joblib.load(path)
    return data[list(data.keys())[0]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args()

    m = load_pkl(args.src)
    s = slice(args.start, args.end)
    root_rot_xyzw = np.asarray(m["root_rot"], dtype=np.float32)[s]
    np.savez(args.dst,
             fps=np.float32(m["fps"]),
             dof=np.asarray(m["dof"], dtype=np.float32)[s],
             root_pos=np.asarray(m["root_trans_offset"], dtype=np.float32)[s],
             root_quat=root_rot_xyzw[:, [3, 0, 1, 2]])  # xyzw -> wxyz (Isaac convention)
    print(f"{args.dst}: {root_rot_xyzw.shape[0]} frames @ {float(m['fps']):.0f} fps")


if __name__ == "__main__":
    main()

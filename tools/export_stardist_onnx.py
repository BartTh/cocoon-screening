#!/usr/bin/env python3
"""Export the StarDist model to ONNX, once, so the app need not ship TensorFlow.

Why this exists twice over:

1. Size. The network is a 5.5 MB U-Net; TensorFlow is roughly 500 MB of the 1.4 GB app.
   onnxruntime is about 15 MB and runs the same graph.
2. The Jetson. The deployment runbook ships ONNX and builds a TensorRT engine on the
   device (see 00_core/orin_nano_ota_runbook.md, sections 5 and 16.5). The same file
   serves both, so the edge path stops being a separate problem.

Run it anywhere TensorFlow is installed:

    pip install tensorflow==2.10.1 tf2onnx onnx
    python tools/export_stardist_onnx.py

Writes models/bee_cocoon_stardist/model.onnx and prints the input/output names the
runtime needs. The app reads only the ONNX file, so re-run this after any retraining and
update model.onnx.sha256-of-weights - a stale export silently keeps the old network.
TensorFlow is no longer in requirements.txt; install tensorflow==2.10.1 to run this.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "bee_cocoon_stardist"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(MODEL_DIR / "model.onnx"))
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    import numpy as np
    import tf2onnx
    import tensorflow as tf
    from stardist.models import StarDist2D

    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    print(f"grid={config['grid']} n_rays={config['n_rays']} channels_out={config['n_channel_out']}")

    model = StarDist2D(None, name=MODEL_DIR.name, basedir=str(MODEL_DIR.parent))
    keras_model = model.keras_model

    # Dynamic height and width: tiles vary with the image, and the network is fully
    # convolutional. Both must stay divisible by the grid, which the caller handles.
    spec = (tf.TensorSpec((None, None, None, config["n_channel_in"]), tf.float32, name="input"),)
    proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=args.opset,
                                          output_path=args.out)

    outputs = [o.name for o in proto.graph.output]
    inputs = [i.name for i in proto.graph.input]
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"\nwrote {args.out} ({size_mb:.1f} MB)")
    print(f"  inputs:  {inputs}")
    print(f"  outputs: {outputs}   (StarDist returns prob, dist in that order)")

    # Parity check: the ONNX graph must agree with Keras, or the label maps silently change.
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n(install onnxruntime to run the parity check)")
        return
    rng = np.random.default_rng(0)
    sample = rng.random((1, 256, 256, config["n_channel_in"]), dtype=np.float32)
    expected = [np.asarray(x) for x in keras_model.predict(sample, verbose=0)]
    actual = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"]).run(None, {inputs[0]: sample})
    for name, want, got in zip(outputs, expected, actual):
        difference = float(np.abs(want - got).max())
        print(f"  {name}: max difference {difference:.2e} {'OK' if difference < 1e-4 else 'TOO LARGE'}")


if __name__ == "__main__":
    main()

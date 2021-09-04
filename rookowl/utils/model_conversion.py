""" Model conversion functions: PyTorch -> ONNX -> TensorFlow -> TensorFlow Lite.
"""

import os

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-


def convert_pt_model_to_onnx(pt_model, input_shape, output_onnx_model_file, simplify=True, verbose=True):
    import torch

    input = torch.randn(size=input_shape)
    print(f"Input Shape  : {list(input.shape)}")
    output = pt_model(input)
    print(f"Output Shape : {list(output.shape)}")

    if verbose:
        print(f"Converting to ONNX model...")
    torch.onnx.export(model=pt_model,
                      args=input,
                      f=output_onnx_model_file,
                      verbose=verbose,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["output"])

    if simplify:
        if verbose:
            print("Simplifying the model...")
        import onnxsim
        onnxsim.simplify(output_onnx_model_file)

    if verbose:
        print(f"ONNX model exported: {output_onnx_model_file}")


def convert_onnx_model_to_tf(onnx_model_file, output_tf_model_dir, verbose=True):
    if verbose:
        print(
            f"Converting ONNX model to TensorFlow model: {output_tf_model_dir}")
    command = f"onnx-tf convert -i \"{onnx_model_file}\" -o  \"{output_tf_model_dir}\""
    if verbose:
        print(f"Running command: {command}")
    result = os.system(command)
    assert result == 0, "Command invocation failed."


def convert_tf_model_to_tflite(tf_model_dir, tflite_model_file, verbose=True):
    import tensorflow as tf
    if verbose:
        print(
            f"Converting TensorFlow model to TensorFlow Lite: {tflite_model_file}")

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    tf_lite_model = converter.convert()

    with open(tflite_model_file, 'wb') as f:
        f.write(tf_lite_model)


def convert_pt_model_to_tflite(pt_model, basename, input_shape, output_tflite_model_file):
    onnx_model_file = f"{basename}.onnx"
    tf_model_dir = f"{basename}.pb"

    convert_pt_model_to_onnx(pt_model, input_shape, onnx_model_file)
    convert_onnx_model_to_tf(onnx_model_file, tf_model_dir)
    # os.remove(onnx_model_file)  <- better not to remove anything from the hard drive
    convert_tf_model_to_tflite(tf_model_dir, output_tflite_model_file)
    # os.rmdir(tf_model_dir)  <- non-empty it is / better not to remove anything from the hard drive

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

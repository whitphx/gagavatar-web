"""Writes scale-model.ts, the refiner tests' network: y = 2 x + 1 on a
[1, 4] float32 tensor, as a base64 ONNX model.

    python make_scale_model.py   # needs the onnx package
"""

import base64
from pathlib import Path

import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
graph = helper.make_graph(
    [helper.make_node("Mul", ["x", "two"], ["t"]), helper.make_node("Add", ["t", "one"], ["y"])],
    "scale",
    [x],
    [y],
    [helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0]), helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
encoded = base64.b64encode(model.SerializeToString()).decode()
Path(__file__).with_name("scale-model.ts").write_text(
    "// Written by make_scale_model.py: y = 2 x + 1 on a [1, 4] float32 tensor.\n"
    f"export const SCALE_MODEL_BASE64 = '{encoded}';\n"
)

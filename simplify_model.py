#!/usr/bin/env python3
"""
Simplify RVC ONNX model for inference by removing random ops
"""
import onnx
from onnx import helper, numpy_helper
import numpy as np

def remove_random_ops(model_path, output_path):
    """Remove RandomNormalLike and RandomUniformLike ops from ONNX model"""
    model = onnx.load(model_path)
    graph = model.graph

    # Find nodes to remove
    nodes_to_remove = []
    nodes_to_add = []

    for node in graph.node:
        if node.op_type in ['RandomNormalLike', 'RandomUniformLike']:
            print(f"Found random op: {node.name} ({node.op_type})")
            print(f"  Inputs: {node.input}")
            print(f"  Outputs: {node.output}")

            # Replace with a constant zero tensor
            # We need to trace what shape this should be
            nodes_to_remove.append(node)

            # For RVC, random ops are typically used in dropout
            # In inference mode, we can replace with zeros or identity
            # The shape comes from the input tensor

            if node.input:
                # Create a Mul by 0 to create zeros in the same shape
                zero_const_name = f"{node.name}_zero"
                mul_node_name = f"{node.name}_mul_zero"

                # Add constant zero
                zero_const = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=[zero_const_name],
                    value=helper.make_tensor(
                        name='const_zero',
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[],
                        vals=[0.0]
                    )
                )

                # Mul input by zero to get zeros in same shape
                mul_node = helper.make_node(
                    'Mul',
                    inputs=[node.input[0], zero_const_name],
                    outputs=node.output,
                    name=mul_node_name
                )

                nodes_to_add.extend([zero_const, mul_node])
                print(f"  -> Replacing with Mul by zero")

    # Remove random nodes
    for node in nodes_to_remove:
        graph.node.remove(node)

    # Add replacement nodes
    for node in nodes_to_add:
        graph.node.append(node)

    print(f"\nRemoved {len(nodes_to_remove)} random ops")
    print(f"Added {len(nodes_to_add)} replacement ops")

    # Save simplified model
    onnx.save(model, output_path)
    print(f"\nSaved simplified model to: {output_path}")

    # Verify the model
    try:
        onnx.checker.check_model(output_path)
        print("✓ Model validation passed")
    except Exception as e:
        print(f"✗ Model validation failed: {e}")

if __name__ == "__main__":
    import sys

    input_model = "models/Yukina_v2_merged.onnx"
    output_model = "models/Yukina_v2_simplified.onnx"

    if len(sys.argv) > 1:
        input_model = sys.argv[1]
    if len(sys.argv) > 2:
        output_model = sys.argv[2]

    print(f"Simplifying: {input_model}")
    print(f"Output to:   {output_model}\n")

    remove_random_ops(input_model, output_model)

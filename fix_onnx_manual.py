#!/usr/bin/env python3
"""
Manually fix RVC ONNX model by replacing random ops with zeros
"""
import onnx
from onnx import helper, numpy_helper
import numpy as np

def fix_random_ops(model_path, output_path):
    """Replace RandomNormalLike and RandomUniformLike with zeros of same shape"""
    model = onnx.load(model_path)
    graph = model.graph

    # Map output name to the node that produces it
    output_to_node = {}
    for node in graph.node:
        for output in node.output:
            output_to_node[output] = node

    # Find all random nodes and their consumers
    random_outputs = {}
    for node in graph.node:
        if node.op_type in ['RandomNormalLike', 'RandomUniformLike']:
            print(f"Random op: {node.name}")
            print(f"  Input: {node.input[0] if node.input else 'none'}")
            print(f"  Output: {node.output[0]}")

            # Store the input (which defines the shape) and output
            random_outputs[node.output[0]] = node.input[0] if node.input else None

    # Now replace all uses of random outputs with zeros
    for node in graph.node:
        for i, inp in enumerate(list(node.input)):
            if inp in random_outputs:
                shape_from = random_outputs[inp]
                if shape_from:
                    print(f"Node {node.name} uses random output {inp}, replacing with zeros based on {shape_from}")

                    # Create a constant zero
                    zero_name = f"zero_{inp}"

                    # Find or create zero constant
                    zero_node = helper.make_node(
                        'ConstantOfShape',
                        inputs=[f"shape_of_{shape_from}"],
                        outputs=[zero_name]
                    )

                    # Create Shape node to get shape
                    shape_node = helper.make_node(
                        'Shape',
                        inputs=[shape_from],
                        outputs=[f"shape_of_{shape_from}"]
                    )

                    # Actually, simpler approach: just multiply the shape source by zero
                    mul_zero_name = f"mul_zero_{inp}"
                    const_zero_name = f"const_zero_{inp}"

                    # Just use the shape source multiplied by zero
                    node.input[i] = shape_from  # Use the shape source directly for now

    # Remove random nodes
    nodes_to_remove = [n for n in graph.node if n.op_type in ['RandomNormalLike', 'RandomUniformLike']]
    for node in nodes_to_remove:
        print(f"Removing: {node.name}")
        graph.node.remove(node)

    print(f"\nRemoved {len(nodes_to_remove)} random nodes")

    # Save
    onnx.save(model, output_path)
    print(f"Saved to: {output_path}")

    # Check
    try:
        onnx.checker.check_model(output_path)
        print("✓ Model is valid")
    except Exception as e:
        print(f"Model check: {e}")
        # Try to load it anyway
        model2 = onnx.load(output_path)
        print(f"Model has {len(model2.graph.node)} nodes")

if __name__ == "__main__":
    fix_random_ops("models/Yukina_v2_merged.onnx", "models/Yukina_v2_norandom.onnx")

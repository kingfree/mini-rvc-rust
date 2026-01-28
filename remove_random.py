#!/usr/bin/env python3
"""
Remove RandomNormalLike/RandomUniformLike from ONNX model
In inference mode, these are used for dropout which should be disabled
We replace uses of their outputs with their inputs (zeros effectively)
"""
import onnx

model = onnx.load("models/Yukina_v2_merged.onnx")
graph = model.graph

# Find random nodes
random_map = {}  # output -> input
random_nodes = []

for node in graph.node:
    if node.op_type in ['RandomNormalLike', 'RandomUniformLike']:
        print(f"Found: {node.name} ({node.op_type})")
        print(f"  Input: {node.input[0] if node.input else 'None'}")
        print(f"  Output: {node.output[0]}")

        # Map output to input (the tensor whose shape we copy)
        if node.input:
            random_map[node.output[0]] = node.input[0]
        random_nodes.append(node)

print(f"\nReplacing {len(random_map)} random outputs...")

# Replace all uses of random outputs
replaced_count = 0
for node in graph.node:
    if node in random_nodes:
        continue

    for i in range(len(node.input)):
        if node.input[i] in random_map:
            old_input = node.input[i]
            new_input = random_map[old_input]
            print(f"  {node.name}: {old_input} -> {new_input}")
            node.input[i] = new_input
            replaced_count += 1

# Remove random nodes
for node in random_nodes:
    graph.node.remove(node)

print(f"\nRemoved {len(random_nodes)} nodes")
print(f"Replaced {replaced_count} inputs")

# Save
output_path = "models/Yukina_v2_norandom.onnx"
onnx.save(model, output_path)
print(f"\nSaved to: {output_path}")

# Verify
try:
    onnx.checker.check_model(model)
    print("✓ Model is valid!")
except Exception as e:
    print(f"⚠ Model check: {e}")
    print("  (May still work in practice)")

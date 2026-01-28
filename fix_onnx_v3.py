import onnx

def topological_sort(model):
    # 简单的 Kahn 算法实现拓扑排序
    graph = model.graph
    nodes = list(graph.node)
    
    # 建立依赖关系
    # 输出名 -> 生成该输出的节点
    output_map = {}
    for node in nodes:
        for output in node.output:
            output_map[output] = node
            
    # 建立入度计数
    # 节点 -> 尚未就绪的输入数量
    in_degree = {node.name: 0 for node in nodes}
    # 节点 -> 依赖它的节点列表
    dependents = {node.name: [] for node in nodes}
    
    # 这里的输入包括 Graph 的 Input 和其他节点的 Output
    known_outputs = set(i.name for i in graph.input) | set(i.name for i in graph.initializer)
    
    for node in nodes:
        for input_name in node.input:
            if input_name in output_map:
                parent = output_map[input_name]
                in_degree[node.name] += 1
                dependents[parent.name].append(node)
    
    # 排序
    queue = [node for node in nodes if in_degree[node.name] == 0]
    sorted_nodes = []
    
    while queue:
        # 保持原始相对顺序
        curr = queue.pop(0)
        sorted_nodes.append(curr)
        for dep in dependents[curr.name]:
            in_degree[dep.name] -= 1
            if in_degree[dep.name] == 0:
                queue.append(dep)
                
    if len(sorted_nodes) != len(nodes):
        print("Warning: Circular dependency or incomplete sort!")
        
    # 替换节点
    while len(graph.node) > 0:
        graph.node.pop()
    graph.node.extend(sorted_nodes)
    return model

m_path = '/home/mei/dev/voice-changer/server/model_dir/0/tsukuyomi_v2_40k_e100_simple.onnx'
o_path = '/home/mei/dev/mini-rvc-rust/model_fixed_v3.onnx'

m = onnx.load(m_path)
m_sorted = topological_sort(m)
onnx.save(m_sorted, o_path)
print(f"Model topologically sorted and saved to {o_path}")

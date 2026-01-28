import onnx
from onnx import numpy_helper, TensorProto

def convert_to_fp32(input_path, output_path):
    model = onnx.load(input_path)
    
    # 修改输入
    for input_node in model.graph.input:
        if input_node.type.tensor_type.elem_type == TensorProto.FLOAT16:
            input_node.type.tensor_type.elem_type = TensorProto.FLOAT
            
    # 修改输出
    for output_node in model.graph.output:
        if output_node.type.tensor_type.elem_type == TensorProto.FLOAT16:
            output_node.type.tensor_type.elem_type = TensorProto.FLOAT
            
    # 修改 Initializers
    for initializer in model.graph.initializer:
        if initializer.data_type == TensorProto.FLOAT16:
            data = numpy_helper.to_array(initializer)
            new_initializer = numpy_helper.from_array(data.astype('float32'), name=initializer.name)
            initializer.CopyFrom(new_initializer)
            
    # 修改所有的 Cast 节点
    for node in model.graph.node:
        if node.op_type == 'Cast':
            for attr in node.attribute:
                if attr.name == 'to':
                    # 如果转成 float16 (10)，则改为 float (1)
                    if attr.i == 10:
                        attr.i = 1
                    # 如果从 float16 转其他，保持逻辑
        
        # 处理节点属性中的 Tensor
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                if attr.t.data_type == TensorProto.FLOAT16:
                    data = numpy_helper.to_array(attr.t)
                    attr.t.CopyFrom(numpy_helper.from_array(data.astype('float32'), name=attr.t.name))

    # 处理 Value Info
    for value_info in model.graph.value_info:
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT16:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT

    onnx.save(model, output_path)
    print(f"Model converted to FP32 and saved to {output_path}")

convert_to_fp32('/home/mei/dev/voice-changer/server/model_dir/0/tsukuyomi_v2_40k_e100_simple.onnx', '/home/mei/dev/mini-rvc-rust/model_fp32_v2.onnx')

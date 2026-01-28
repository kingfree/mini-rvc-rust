import onnx
from onnx import numpy_helper

def convert_float16_to_float32(model):
    for i in range(len(model.graph.initializer)):
        tensor = model.graph.initializer[i]
        if tensor.data_type == onnx.TensorProto.FLOAT16:
            data = numpy_helper.to_array(tensor)
            new_tensor = numpy_helper.from_array(data.astype('float32'), name=tensor.name)
            model.graph.initializer[i].CopyFrom(new_tensor)

    for i in range(len(model.graph.input)):
        if model.graph.input[i].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            model.graph.input[i].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            
    for i in range(len(model.graph.output)):
        if model.graph.output[i].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            model.graph.output[i].type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    for i in range(len(model.graph.value_info)):
        if model.graph.value_info[i].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            model.graph.value_info[i].type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                if attribute.t.data_type == onnx.TensorProto.FLOAT16:
                    data = numpy_helper.to_array(attribute.t)
                    attribute.t.CopyFrom(numpy_helper.from_array(data.astype('float32'), name=attribute.t.name))
    return model

model = onnx.load('/home/mei/dev/voice-changer/server/model_dir/0/tsukuyomi_v2_40k_e100_simple.onnx')
model = convert_float16_to_float32(model)
onnx.save(model, '/home/mei/dev/mini-rvc-rust/model_fp32.onnx')
print("Conversion complete")

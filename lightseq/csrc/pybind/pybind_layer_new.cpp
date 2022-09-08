#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <string>

#include "context.h"
#include "cuda_util.h"
#include "transformer_encoder_layer.h"

// x is torch::Tensor
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace lightseq {

template <typename T>
const T *rptr(const torch::Tensor &tensor) {
  return reinterpret_cast<const T *>(tensor.data_ptr());
}

template <typename T>
T *rptr(torch::Tensor &tensor) {
  return reinterpret_cast<T *>(tensor.data_ptr());
}

static std::unordered_map<int, std::shared_ptr<void>>
    s_transformer_encoder_layers;

void ContextInitial() {
  static ContextPtr context_ptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (context_ptr == nullptr) {
    context_ptr.reset(new Context(true));
    Context::set_thread_context(context_ptr);
  }
  context_ptr->set_stream(stream);
}

template <typename T1, typename T2>
int create_transformer_encoder_layer_new(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_dim,
    int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, float hidden_dropout_ratio,
    bool pre_or_postLayerNorm, std::string activation_fn,
    bool mask_future_tokens) {
  // necessary
  ContextInitial();

  auto layer = std::make_shared<TransformerEncoderLayer<T1, T2>>(
      layer_id, max_batch_tokens, max_seq_len, hidden_dim, num_heads,
      intermediate_size, attn_prob_dropout_ratio, activation_dropout_ratio,
      hidden_dropout_ratio, pre_or_postLayerNorm, activation_fn,
      mask_future_tokens);

  // layer->load_para_and_grad(rptr<T1>(para_ptr), rptr<T2>(grad_ptr));

  Variable *inp(new Variable("input"));
  Variable *inp_mask(new Variable("inp_mask"));

  Variable *layer_out = (*layer)(inp, inp_mask);

  s_transformer_encoder_layers[layer_id] = layer;

  const int default_batch_size = 1;
  const int default_seq_len = 64;
  layer->before_forward(default_batch_size, default_seq_len);

  std::string T1_dtype = (std::is_same<T1, __half>::value) ? "half" : "float";
  std::string T2_dtype = (std::is_same<T2, __half>::value) ? "half" : "float";

  std::cout << "Encoder layer #" << layer_id << " is created with date type ["
            << T1_dtype << ", " << T2_dtype << "]." << std::endl;

  return 0;
}

template <typename T1, typename T2>
std::vector<torch::Tensor> transformer_encoder_layer_fw(
    int layer_id, const torch::Tensor &input, const torch::Tensor &input_mask,
    bool training_mode) {
  CHECK_INPUT(input);
  CHECK_INPUT(input_mask);

  auto output = torch::empty_like(input);

  const char *input_ptr = (const char *)input.data_ptr();
  const char *input_mask_ptr = (const char *)input_mask.data_ptr();

  char *out_ptr = (char *)output.data_ptr();

  std::shared_ptr<TransformerEncoderLayer<T1, T2>> layer =
      std::static_pointer_cast<TransformerEncoderLayer<T1, T2>>(
          s_transformer_encoder_layers[layer_id]);

  Variable *inp_node = layer->input(0);
  inp_node->set_value(input_ptr);
  Variable *inp_mask_node = layer->input(1);
  inp_mask_node->set_value(input_mask_ptr);

  Variable *out_node = layer->output(0);
  out_node->set_value(out_ptr);

  layer->before_forward(input.size(0), input.size(1));

  layer->forward();

  return {output};
}

template <typename T1, typename T2>
std::vector<torch::Tensor> transformer_encoder_layer_bw(
    int layer_id, const torch::Tensor &grad_out, const torch::Tensor &output,
    const torch::Tensor &input, const torch::Tensor &input_mask) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(output);
  CHECK_INPUT(input);
  CHECK_INPUT(input_mask);

  auto grad_inp = torch::empty_like(grad_out);

  // inputs.
  char *grad_output_ptr = (char *)grad_out.data_ptr();
  const char *input_ptr = (const char *)input.data_ptr();
  const char *output_ptr = (const char *)output.data_ptr();
  const char *input_mask_ptr = (const char *)input_mask.data_ptr();

  // outputs.
  char *grad_input_ptr = (char *)grad_inp.data_ptr();

  std::shared_ptr<TransformerEncoderLayer<T1, T2>> layer =
      std::static_pointer_cast<TransformerEncoderLayer<T1, T2>>(
          s_transformer_encoder_layers[layer_id]);

  Variable *inp_node = layer->input(0);
  inp_node->set_value(input_ptr);
  inp_node->set_grad(grad_input_ptr);
  Variable *inp_mask_node = layer->input(1);
  inp_mask_node->set_value(input_mask_ptr);

  Variable *out_node = layer->output(0);
  out_node->set_value(output_ptr);
  out_node->set_grad(grad_output_ptr);

  layer->backward();

  return {grad_inp};
}

template <typename T1, typename T2>
void assign_layer_weight_grad(const torch::Tensor &weights,
                              torch::Tensor &grads, std::string layer_name,
                              int layer_id) {
  CHECK_INPUT(weights);
  const T1 *wptr = (const T1 *)weights.data_ptr();

  CHECK_INPUT(grads);
  T2 *gptr = (T2 *)grads.data_ptr();

  if (layer_name == "TransformerEncoderLayer") {
    std::shared_ptr<TransformerEncoderLayer<T1, T2>> layer =
        std::static_pointer_cast<TransformerEncoderLayer<T1, T2>>(
            s_transformer_encoder_layers[layer_id]);
    layer->load_para_and_grad(wptr, gptr);
  } else {
    printf("Error! layer_name %s is unsupported!\n", layer_name.c_str());
    exit(-1);
  }
  std::cout << layer_name << " #" << layer_id << " bind weights and grads."
            << std::endl;
  return;
}

}  // namespace lightseq

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_transformer_encoder_layer_new_fp32",
        &lightseq::create_transformer_encoder_layer_new<float, float>,
        "Create LightSeq Transformer Encoder Layer with fp32 (CUDA)");
  m.def("create_transformer_encoder_layer_new_fp16",
        &lightseq::create_transformer_encoder_layer_new<__half, __half>,
        "Create LightSeq Transformer Encoder Layer with fp16 (CUDA)");

  m.def("transformer_encoder_layer_fw_fp32",
        &lightseq::transformer_encoder_layer_fw<float, float>,
        "LightSeq Transformer Encoder forward with fp32 (CUDA)");
  m.def("transformer_encoder_layer_fw_fp16",
        &lightseq::transformer_encoder_layer_fw<__half, __half>,
        "LightSeq Transformer Encoder forward with fp16 (CUDA)");

  m.def("transformer_encoder_layer_bw_fp32",
        &lightseq::transformer_encoder_layer_bw<float, float>,
        "LightSeq Transformer Encoder forward with fp32 (CUDA)");
  m.def("transformer_encoder_layer_bw_fp16",
        &lightseq::transformer_encoder_layer_bw<__half, __half>,
        "LightSeq Transformer Encoder forward with fp16 (CUDA)");

  m.def("assign_layer_weight_grad_fp32",
        &lightseq::assign_layer_weight_grad<float, float>,
        "Bind layer weights and grads");
  m.def("assign_layer_weight_grad_fp16",
        &lightseq::assign_layer_weight_grad<__half, __half>,
        "Bind layer weights and grads");
}
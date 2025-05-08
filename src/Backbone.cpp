#include "Backbone.hpp"
using namespace backbone;

// https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py 
// https://tech.bertelsmann.com/en/blog/articles/convnext 

// ---------------------------------------------------------------- ConvNeXt ---------------------------------------------------------------- // 
ConvNeXtBlockImpl::ConvNeXtBlockImpl::ConvNeXtBlockImpl(int in_channels, float drop_path, float layer_scale_init_value):
    dwconv(torch::nn::Conv2dOptions(in_channels, in_channels, 7).padding(3).groups(in_channels)),
    norm(torch::nn::LayerNormOptions({in_channels}).elementwise_affine(true)),
    pwconv1(in_channels, 4 * in_channels),
    pwconv2(4 * in_channels, in_channels),
    drop_path_rate(drop_path)
{
    register_module("dwconv", dwconv);
    register_module("norm", norm);
    register_module("pwconv1", pwconv1);
    register_module("pwconv2", pwconv2);

    // Initialize gamma (Layer Scale)
    if (layer_scale_init_value > 0) {
        gamma = register_parameter("gamma", torch::full({in_channels}, layer_scale_init_value));
    }

    // DropPath is only applied during training
    if (drop_path > 0) {
        drop_path_enabled = true;
    }

    // this->apply([](torch::nn::Module& module) {
    //     if (auto* conv = module.as<torch::nn::Conv2d>()) {
    //         torch::nn::init::kaiming_normal_(
    //             conv->weight, 0.0, torch::kFanOut, torch::kReLU);
    //     } else if (auto* linear = module.as<torch::nn::Linear>()) {
    //         torch::nn::init::trunc_normal_(linear->weight, 0.02);
    //         if (linear->bias.defined())
    //             torch::nn::init::zeros_(linear->bias);
    //     }
    // });
}

torch::Tensor ConvNeXtBlockImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x.clone();

    x = dwconv->forward(x);
    x = x.permute({0, 2, 3, 1}); // (N, C, H, W) -> (N, H, W, C)
    x = norm->forward(x);
    x = pwconv1->forward(x);
    x = torch::gelu(x);
    x = pwconv2->forward(x);
    if (gamma.defined()) {
        // x = x * gamma;
        // x = x * gamma.unsqueeze(0).unsqueeze(0).unsqueeze(0);
        x = x * gamma.view({1, 1, 1, -1});
    }
    x = x.permute({0, 3, 1, 2}); // (N, H, W, C) -> (N, C, H, W)
    // Apply DropPath
    if (drop_path_enabled && is_training()) {
        x = drop_path(x, drop_path_rate);
    }
    x = residual + x;
    return x;
}
// Solution adapted from : https://github.com/FrancescoSaverioZuppichini/DropPath 
torch::Tensor ConvNeXtBlockImpl::drop_path(torch::Tensor x, float drop_prob){
    if (drop_prob == 0.0 || !x.requires_grad()) {
        return x;
    }
    auto batch_size = x.size(0);
    auto shape = std::vector<int64_t>({batch_size, 1, 1, 1});  // Broadcast over (N, C, H, W)
    
    auto keep_prob = 1.0f - drop_prob;
    auto random_tensor = torch::empty(shape, x.options()).uniform_(0, 1);  // uniform in [0, 1]
    random_tensor = (random_tensor < keep_prob).to(x.dtype());  // binary mask: 1 = keep, 0 = drop

    // Scale to preserve expected value during training
    return x / keep_prob * random_tensor;

}


// Solution adapted from : https://github.com/wilile26811249/MobileViT/blob/main/models/module.py 
// ---------------------------------------------------------------- MobileVit ---------------------------------------------------------------- // 
ConvNormActImpl::ConvNormActImpl(int in_channels, int out_channels, int kernel_size, int stride, int padding, int groups, bool use_bn, bool use_act)
    : use_bn(use_bn), use_act(use_act) {
    if (padding == -1)
        padding = (kernel_size - 1) / 2;
    
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                                          .stride(stride)
                                                          .padding(padding)
                                                          .groups(groups)
                                                          .bias(!use_bn)));
    
    if (use_bn)
        bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
    if (use_act)
        act = register_module("act", torch::nn::SiLU());
}

torch::Tensor ConvNormActImpl::forward(torch::Tensor x) {
    x = conv->forward(x);
    if (use_bn) x = bn->forward(x);
    if (use_act) x = act->forward(x);
    return x;
}

// MultiHeadSelfAttention
MultiHeadSelfAttentionImpl::MultiHeadSelfAttentionImpl(int64_t dim, int64_t num_heads) 
    : num_heads(num_heads), dim_head(dim / num_heads), scale_factor(1.0f / std::sqrt(dim)) {
    
    int64_t weight_dim = num_heads * dim_head;
    to_qvk = register_module("to_qvk", torch::nn::Linear(torch::nn::LinearOptions(dim, weight_dim * 3).bias(false)));
    w_out = register_module("w_out", torch::nn::Linear(torch::nn::LinearOptions(weight_dim, dim).bias(false)));
}

torch::Tensor MultiHeadSelfAttentionImpl::forward(torch::Tensor x) {
    // Print input shape for debugging
    // std::cout << "MHSA input shape: " << x.sizes() << std::endl;
    
    // Get original shape
    int64_t b = x.size(0);  // batch
    int64_t np = x.size(1); // number of patches
    int64_t nt = x.size(2); // number of tokens per patch
    int64_t d = x.size(3);  // embedding dimension
    
    // Validate dimensions
    if (d != to_qvk->weight.size(1)) {
        std::cerr << "Input dimension " << d << " doesn't match weight dimension " << to_qvk->weight.size(1) << std::endl;
        throw std::runtime_error("Dimension mismatch in MultiHeadSelfAttention");
    }

    // std::cout << "Reshaping tensor from [" << b << ", " << np << ", " << nt << ", " << d << "] to [" 
    //           << b << ", " << np * nt << ", " << d << "]" << std::endl;
    
    // Reshape to [batch, sequence_length, dimension] for the linear projection
    auto x_flat = x.reshape({b, np * nt, d});

    // std::cout << "x_flat shape: " << x_flat.sizes() << std::endl;
    // std::cout << "to_qvk weight shape: " << to_qvk->weight.sizes() << std::endl;
    
    // Compute QKV projections - handle potential errors
    torch::Tensor qkv;
    try {
        qkv = to_qvk->forward(x_flat);
        // std::cout << "QKV shape: " << qkv.sizes() << std::endl;
    } catch (const c10::Error& e) {
        // std::cerr << "Error in QKV projection: " << e.what() << std::endl;
        throw;
    }
    
    auto qkv_chunks = qkv.chunk(3, -1);
    
    // Get projected dimension
    int64_t proj_dim = qkv_chunks[0].size(-1);
    // std::cout << "Projected dimension: " << proj_dim << ", expected: " << num_heads * dim_head << std::endl;
    
    // Reshape to [batch, heads, sequence_length, dim_head]
    try {
        auto q = qkv_chunks[0].reshape({b, np * nt, num_heads, dim_head}).permute({0, 2, 1, 3});
        auto k = qkv_chunks[1].reshape({b, np * nt, num_heads, dim_head}).permute({0, 2, 1, 3});
        auto v = qkv_chunks[2].reshape({b, np * nt, num_heads, dim_head}).permute({0, 2, 1, 3});
        
        // std::cout << "Q shape: " << q.sizes() << std::endl;
        // std::cout << "K shape: " << k.sizes() << std::endl;
        
        // Compute attention scores and apply attention
        auto attn = torch::matmul(q, k.transpose(-1, -2));
        // std::cout << "Raw attention shape: " << attn.sizes() << std::endl;
        
        attn = attn * scale_factor;
        attn = torch::softmax(attn, -1);
        
        // Matrix multiply with values
        auto out = torch::matmul(attn, v);
        // std::cout << "Attention output shape: " << out.sizes() << std::endl;
        out = out.permute({0, 2, 1, 3}).reshape({b, np * nt, num_heads * dim_head});
        out = w_out->forward(out);
        
        // Reshape to original format
        return out.reshape({b, np, nt, d});
    } catch (const c10::Error& e) {
        std::cerr << "Error in attention computation: " << e.what() << std::endl;
        throw;
    }
}

// FFN
FFNImpl::FFNImpl(int64_t dim, int64_t hidden_dim, double dropout) {
    net = register_module("net", torch::nn::Sequential(
        torch::nn::Linear(dim, hidden_dim),
        torch::nn::SiLU(),
        torch::nn::Dropout(dropout),
        torch::nn::Linear(hidden_dim, dim),
        torch::nn::Dropout(dropout)
    ));
}

torch::Tensor FFNImpl::forward(torch::Tensor x) {
    // std::cout << "ERROR from here" << std::endl;
    return net->forward(x);
}

// Transformer
TransformerImpl::TransformerImpl(int dim, int depth, int heads, int dim_head, int mlp_dim, double dropout) {
    for (int i = 0; i < depth; ++i) {
        layers->push_back(torch::nn::Sequential(
            MultiHeadSelfAttention(dim, heads),
            FFN(dim, mlp_dim, dropout)
        ));
    }
    register_module("layers", layers);
}

torch::Tensor TransformerImpl::forward(torch::Tensor x) {
    for (auto& layer : *layers) {
        x = x + layer->as<torch::nn::Sequential>()->forward(x);
    }
    return x;
}

// InvertedResidual
InvertedResidualImpl::InvertedResidualImpl(int in_channels, int out_channels, int stride, float expand_ratio)
    : use_res_connect(stride == 1 && in_channels == out_channels) {
    
    int hidden_dim = static_cast<int>(in_channels * expand_ratio);

    torch::nn::Sequential layers;
    if (expand_ratio != 1.0f)
        layers->push_back(ConvNormAct(in_channels, hidden_dim, 1, 1, 0));

    layers->push_back(ConvNormAct(hidden_dim, hidden_dim, 3, stride, 1, hidden_dim));

    layers->push_back(register_module("conv2d", torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_dim, out_channels, 1))));  

    layers->push_back(register_module("batch_norm", torch::nn::BatchNorm2d(out_channels)));

    conv = register_module("conv", layers);
}

torch::Tensor InvertedResidualImpl::forward(torch::Tensor x) {
    return use_res_connect ? x + conv->forward(x) : conv->forward(x);
}

// MobileVitBlock Constructor
MobileVitBlockImpl::MobileVitBlockImpl(int in_channels, int out_channels, int d_model, int layers, int mlp_dim)
    {
    local_representation = register_module("local_representation", torch::nn::Sequential(
            ConvNormAct(in_channels, in_channels, 3), // Local spatial encoding
            ConvNormAct(in_channels, d_model, 1) 
        ));

    transformer = register_module("transformer", Transformer(d_model, layers, 1, 32, mlp_dim));
    
    fusion_block1 = register_module("fusion_block1", torch::nn::Conv2d(torch::nn::Conv2dOptions(d_model, in_channels, 1)));
    fusion_block2 = register_module("fusion_block2", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels * 2, out_channels, 3).stride(1).padding(1))); 
}
// Forward Pass
torch::Tensor MobileVitBlockImpl::forward(torch::Tensor x) {
    auto local_repr = local_representation->forward(x);
    
    // Extract shape
    int b = local_repr.size(0);  // Batch size
    int d = local_repr.size(1);  // Channel dimension (d_model)
    int h = local_repr.size(2);  // Height
    int w = local_repr.size(3);  // Width

    // Rearrange to (batch, num_patches, num_tokens, channels) for transformer input
    int ph = 2, pw = 2;  // Patch size
    // std::cout << "local_repr shape: " << local_repr.sizes() << std::endl;
    // std::cout << "Batch: " << b << ", Channels: " << d << ", Height: " << h << ", Width: " << w << std::endl;
    // std::cout << "Reshaping into: (" << b << ", " << d << ", " << h/ph << ", " << ph << ", " << w/pw << ", " << pw << ")" << std::endl;
    if (h % ph != 0 || w % pw != 0) {
        throw std::runtime_error("Height and width must be divisible by patch size!");
    }
    auto global_repr = local_repr.view({b, d, h / ph, ph, w / pw, pw})  // Flatten height and width into patches
                             .permute({0, 2, 4, 1, 3, 5})          // Reorder dimensions: batch, height patches, width patches, channels, patch height, patch width
                             .contiguous()
                             .view({b, (h / ph) * (w / pw), ph * pw, d});  // Flatten to (batch, num_patches, num_tokens, channels)

    
    std::cout << "global_repr shape before transformer: " << global_repr.sizes() << std::endl;
    if (torch::any(torch::isnan(global_repr)).item<bool>()) {
        std::cerr << "global_repr contains NaN!" << std::endl;
    }
    if (torch::any(torch::isinf(global_repr)).item<bool>()) {
        std::cerr << "global_repr contains Inf!" << std::endl;
    }
    
    // Apply Transformer
    global_repr = transformer->forward(global_repr);
    std::cout << "global_repr shape after transformer: " << global_repr.sizes() << std::endl;
    // Rearrange back to (batch, d_model, H, W)
    global_repr = global_repr.view({b, h / ph, w / pw, ph, pw, d})
                             .permute({0, 5, 1, 3, 2, 4})
                             .contiguous()
                             .view({b, d, h, w});
    std::cout << "global_repr shape after transformer and reshaping: " << global_repr.sizes() << std::endl;
    if (torch::any(torch::isnan(global_repr)).item<bool>()) {
        std::cerr << "global_repr contains NaN!" << std::endl;
    }
    if (torch::any(torch::isinf(global_repr)).item<bool>()) {
        std::cerr << "global_repr contains Inf!" << std::endl;
    }
    
    // Fuse Local & Global Representations
    auto fuse_repr = fusion_block1->forward(global_repr);
    std::cout << "fuse_repr" << std::endl;
    auto result = fusion_block2->forward(torch::cat({x, fuse_repr}, 1));
    std::cout << "result" << std::endl;
    return result;
}

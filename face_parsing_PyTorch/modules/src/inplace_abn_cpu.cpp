#include <ATen/ATen.h>

#include <vector>

#include "utils/checks.h"
#include "inplace_abn.h"

at::Tensor reduce_sum(at::Tensor x) {
  if (x.ndimension() == 2) {
    return x.sum(0);
  } else {
    auto x_view = x.view({x.size(0), x.size(1), -1});
    return x_view.sum(-1).sum(0);
  }
}

at::Tensor broadcast_to(at::Tensor v, at::Tensor x) {
  if (x.ndimension() == 2) {
    return v;
  } else {
    std::vector<int64_t> broadcast_size = {1, -1};
    for (int64_t i = 2; i < x.ndimension(); ++i)
      broadcast_size.push_back(1);

    return v.view(broadcast_size);
  }
}

int64_t count(at::Tensor x) {
  int64_t count = x.size(0);
  for (int64_t i = 2; i < x.ndimension(); ++i)
    count *= x.size(i);

  return count;
}

at::Tensor invert_affine(at::Tensor z, at::Tensor weight, at::Tensor bias, bool affine, float eps) {
  if (affine) {
    return (z - broadcast_to(bias, z)) / broadcast_to(at::abs(weight) + eps, z);
  } else {
    return z;
  }
}

std::vector<at::Tensor> mean_var_cpu(at::Tensor x) {
  auto num = count(x);
  auto mean = reduce_sum(x) / num;
  auto diff = x - broadcast_to(mean, x);
  auto var = reduce_sum(diff.pow(2)) / num;

  return {mean, var};
}

at::Tensor forward_cpu(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
                       bool affine, float eps) {
  auto gamma = affine ? at::abs(weight) + eps : at::ones_like(var);
  auto mul = at::rsqrt(var + eps) * gamma;

  x.sub_(broadcast_to(mean, x));
  x.mul_(broadcast_to(mul, x));
  if (affine) x.add_(broadcast_to(bias, x));

  return x;
}

std::vector<at::Tensor> edz_eydz_cpu(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
                                     bool affine, float eps) {
  auto edz = reduce_sum(dz);
  auto y = invert_affine(z, weight, bias, affine, eps);
  auto eydz = reduce_sum(y * dz);

  return {edz, eydz};
}

at::Tensor backward_cpu(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
                                     at::Tensor edz, at::Tensor eydz, bool affine, float eps) {
  auto y = invert_affine(z, weight, bias, affine, eps);
  auto mul = affine ? at::rsqrt(var + eps) * (at::abs(weight) + eps) : at::rsqrt(var + eps);

  auto num = count(z);
  auto dx = (dz - broadcast_to(edz / num, dz) - y * broadcast_to(eydz / num, dz)) * broadcast_to(mul, dz);
  return dx;
}

void leaky_relu_backward_cpu(at::Tensor z, at::Tensor dz, float slope) {
  CHECK_CPU_INPUT(z);
  CHECK_CPU_INPUT(dz);

  AT_DISPATCH_FLOATING_TYPES(z.type(), "leaky_relu_backward_cpu", ([&] {
    int64_t count = z.numel();
    auto *_z = z.data<scalar_t>();
    auto *_dz = dz.data<scalar_t>();

    for (int64_t i = 0; i < count; ++i) {
      if (_z[i] < 0) {
        _z[i] *= 1 / slope;
        _dz[i] *= slope;
      }
    }
  }));
}

void elu_backward_cpu(at::Tensor z, at::Tensor dz) {
  CHECK_CPU_INPUT(z);
  CHECK_CPU_INPUT(dz);

  AT_DISPATCH_FLOATING_TYPES(z.type(), "elu_backward_cpu", ([&] {
    int64_t count = z.numel();
    auto *_z = z.data<scalar_t>();
    auto *_dz = dz.data<scalar_t>();

    for (int64_t i = 0; i < count; ++i) {
      if (_z[i] < 0) {
        _z[i] = log1p(_z[i]);
        _dz[i] *= (_z[i] + 1.f);
      }
    }
  }));
}

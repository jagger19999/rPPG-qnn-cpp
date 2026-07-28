#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rppg_qnn {

struct YuvPlaneView {
  const std::uint8_t* data;
  std::size_t size;
  int row_stride;
  int pixel_stride;
};

struct Yuv420View {
  int width;
  int height;
  YuvPlaneView y;
  YuvPlaneView u;
  YuvPlaneView v;
};

enum class YuvMatrix { Bt601, Bt709 };
enum class YuvRange { Limited, Full };
enum class ChromaOrder { Uv, Vu };

struct YuvColorSpec {
  YuvMatrix matrix{YuvMatrix::Bt601};
  YuvRange range{YuvRange::Limited};
  ChromaOrder chroma_order{ChromaOrder::Uv};
};

std::vector<std::uint8_t> yuv420_to_bgr(const Yuv420View& image);
std::vector<std::uint8_t> yuv420_to_bgr(const Yuv420View& image,
                                        YuvColorSpec spec);

}  // namespace rppg_qnn

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

std::vector<std::uint8_t> yuv420_to_bgr(const Yuv420View& image);

}  // namespace rppg_qnn

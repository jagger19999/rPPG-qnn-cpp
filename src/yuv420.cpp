#include "rppg_qnn/yuv420.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace rppg_qnn {
namespace {

std::size_t output_size(int width, int height) {
  const auto converted_width = static_cast<std::size_t>(width);
  const auto converted_height = static_cast<std::size_t>(height);
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (converted_width > maximum / converted_height) {
    throw std::overflow_error("YUV image dimensions overflow");
  }
  const std::size_t pixels = converted_width * converted_height;
  if (pixels > maximum / 3U) {
    throw std::overflow_error("BGR output size overflows");
  }
  return pixels * 3U;
}

void validate_plane(const YuvPlaneView& plane, int width, int height) {
  if (plane.data == nullptr || plane.row_stride <= 0 ||
      (plane.pixel_stride != 1 && plane.pixel_stride != 2)) {
    throw std::invalid_argument("invalid YUV plane layout");
  }

  const auto rows = static_cast<std::size_t>(height);
  const auto columns = static_cast<std::size_t>(width);
  const auto row_stride = static_cast<std::size_t>(plane.row_stride);
  const auto pixel_stride = static_cast<std::size_t>(plane.pixel_stride);
  const auto maximum = std::numeric_limits<std::size_t>::max();
  const std::size_t last_row = rows - 1U;
  const std::size_t last_column = columns - 1U;

  if (last_column > maximum / pixel_stride) {
    throw std::invalid_argument("YUV plane offset overflows");
  }
  const std::size_t column_offset = last_column * pixel_stride;
  if (last_row > (maximum - column_offset) / row_stride) {
    throw std::invalid_argument("YUV plane offset overflows");
  }
  const std::size_t last_offset = last_row * row_stride + column_offset;
  if (last_offset >= plane.size) {
    throw std::invalid_argument("YUV plane is truncated");
  }
}

int divide_by_256_floor(int value) {
  if (value >= 0) {
    return value / 256;
  }
  return -((-value + 255) / 256);
}

std::uint8_t clamp_byte(int value) {
  return static_cast<std::uint8_t>(std::clamp(value, 0, 255));
}

}  // namespace

std::vector<std::uint8_t> yuv420_to_bgr(const Yuv420View& image) {
  if (image.width <= 0 || image.height <= 0) {
    throw std::invalid_argument("YUV image dimensions must be positive");
  }

  const std::size_t bgr_size = output_size(image.width, image.height);
  const int chroma_width = image.width / 2 + image.width % 2;
  const int chroma_height = image.height / 2 + image.height % 2;
  validate_plane(image.y, image.width, image.height);
  validate_plane(image.u, chroma_width, chroma_height);
  validate_plane(image.v, chroma_width, chroma_height);

  std::vector<std::uint8_t> bgr(bgr_size);
  std::size_t destination = 0;
  for (int row = 0; row < image.height; ++row) {
    const auto y_row = static_cast<std::size_t>(row) *
                       static_cast<std::size_t>(image.y.row_stride);
    const auto chroma_row = static_cast<std::size_t>(row / 2);
    const auto u_row = chroma_row * static_cast<std::size_t>(image.u.row_stride);
    const auto v_row = chroma_row * static_cast<std::size_t>(image.v.row_stride);
    for (int column = 0; column < image.width; ++column) {
      const auto y_offset =
          y_row + static_cast<std::size_t>(column) *
                      static_cast<std::size_t>(image.y.pixel_stride);
      const auto chroma_column = static_cast<std::size_t>(column / 2);
      const auto u_offset =
          u_row + chroma_column * static_cast<std::size_t>(image.u.pixel_stride);
      const auto v_offset =
          v_row + chroma_column * static_cast<std::size_t>(image.v.pixel_stride);

      const int c = static_cast<int>(image.y.data[y_offset]) - 16;
      const int d = static_cast<int>(image.u.data[u_offset]) - 128;
      const int e = static_cast<int>(image.v.data[v_offset]) - 128;
      const int red = divide_by_256_floor(298 * c + 409 * e + 128);
      const int green =
          divide_by_256_floor(298 * c - 100 * d - 208 * e + 128);
      const int blue = divide_by_256_floor(298 * c + 516 * d + 128);

      bgr[destination++] = clamp_byte(blue);
      bgr[destination++] = clamp_byte(green);
      bgr[destination++] = clamp_byte(red);
    }
  }
  return bgr;
}

}  // namespace rppg_qnn

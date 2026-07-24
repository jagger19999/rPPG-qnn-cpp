#include "rppg_qnn/yuv420.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "test_support.hpp"

namespace {

using rppg_qnn::Yuv420View;
using rppg_qnn::YuvPlaneView;

template <typename Exception, typename Operation>
void expect_throws(Operation&& operation) {
  try {
    operation();
  } catch (const Exception&) {
    return;
  } catch (...) {
    EXPECT_TRUE(false);
    return;
  }
  EXPECT_TRUE(false);
}

YuvPlaneView plane(const std::vector<std::uint8_t>& bytes,
                   int row_stride,
                   int pixel_stride) {
  return {bytes.data(), bytes.size(), row_stride, pixel_stride};
}

void converts_contiguous_planar_4x2_to_exact_bgr() {
  const std::vector<std::uint8_t> y = {16, 82, 145, 235, 41, 210, 100, 180};
  const std::vector<std::uint8_t> u = {128, 90};
  const std::vector<std::uint8_t> v = {128, 240};
  const Yuv420View image{4, 2, plane(y, 4, 1), plane(u, 2, 1), plane(v, 2, 1)};

  const std::vector<std::uint8_t> expected = {
      0,   0,   0,   77,  77,  77,  74,  74,  255, 178, 179, 255,
      29,  29,  29,  226, 226, 226, 21,  22,  255, 114, 115, 255,
  };
  EXPECT_EQ(rppg_qnn::yuv420_to_bgr(image), expected);
}

void ignores_padding_at_the_end_of_plane_rows() {
  const std::vector<std::uint8_t> y = {
      16, 82, 145, 235, 7, 7, 41, 210, 100, 180, 7, 7,
      60, 120, 170, 220, 7, 7, 30, 90, 150, 200, 7, 7,
  };
  const std::vector<std::uint8_t> u = {128, 90, 7, 7, 240, 54, 7, 7};
  const std::vector<std::uint8_t> v = {128, 240, 7, 7, 110, 34, 7, 7};
  const Yuv420View image{4, 4, plane(y, 6, 1), plane(u, 4, 1), plane(v, 4, 1)};

  const std::vector<std::uint8_t> expected = {
      0,   0,   0,   77,  77,  77,  74,  74,  255, 178, 179, 255,
      29,  29,  29,  226, 226, 226, 21,  22,  255, 114, 115, 255,
      255, 22,  22,  255, 92,  92,  30,  255, 29,  88,  255, 87,
      242, 0,   0,   255, 57,  57,  7,   255, 6,   65,  255, 64,
  };
  EXPECT_EQ(rppg_qnn::yuv420_to_bgr(image), expected);
}

void supports_pixel_stride_two_interleaved_chroma_views() {
  const std::vector<std::uint8_t> y = {16, 82, 145, 235, 41, 210, 100, 180};
  const std::vector<std::uint8_t> uv = {90, 240, 128, 128};
  const YuvPlaneView u{uv.data(), uv.size(), 4, 2};
  const YuvPlaneView v{uv.data() + 1, uv.size() - 1, 4, 2};
  const Yuv420View image{4, 2, plane(y, 4, 1), u, v};

  const std::vector<std::uint8_t> expected = {
      0,   0,   179, 0,   1,   255, 150, 150, 150, 255, 255, 255,
      0,   0,   208, 149, 150, 255, 98,  98,  98,  191, 191, 191,
  };
  EXPECT_EQ(rppg_qnn::yuv420_to_bgr(image), expected);
}

void converts_odd_3x3_dimensions_using_ceil_chroma_extent() {
  const std::vector<std::uint8_t> y = {16, 41, 82, 100, 145, 180, 210, 235, 128};
  const std::vector<std::uint8_t> u = {128, 128, 128, 128};
  const std::vector<std::uint8_t> v = {128, 128, 128, 128};
  const Yuv420View image{3, 3, plane(y, 3, 1), plane(u, 2, 1), plane(v, 2, 1)};

  const std::vector<std::uint8_t> expected = {
      0,   0,   0,   29,  29,  29,  77,  77,  77,
      98,  98,  98,  150, 150, 150, 191, 191, 191,
      226, 226, 226, 255, 255, 255, 130, 130, 130,
  };
  EXPECT_EQ(rppg_qnn::yuv420_to_bgr(image), expected);
}

void rejects_truncated_planes() {
  const std::vector<std::uint8_t> y(9, 128);
  const std::vector<std::uint8_t> chroma(4, 128);

  expect_throws<std::invalid_argument>([&]() {
    const Yuv420View image{3, 3, {y.data(), 8, 3, 1},
                           plane(chroma, 2, 1), plane(chroma, 2, 1)};
    (void)rppg_qnn::yuv420_to_bgr(image);
  });
  expect_throws<std::invalid_argument>([&]() {
    const Yuv420View image{3, 3, plane(y, 3, 1),
                           {chroma.data(), 3, 2, 1}, plane(chroma, 2, 1)};
    (void)rppg_qnn::yuv420_to_bgr(image);
  });
  expect_throws<std::invalid_argument>([&]() {
    const Yuv420View image{3, 3, plane(y, 3, 1),
                           plane(chroma, 2, 1), {chroma.data(), 3, 2, 1}};
    (void)rppg_qnn::yuv420_to_bgr(image);
  });
}

void rejects_invalid_strides_and_dimensions() {
  const std::vector<std::uint8_t> y(4, 128);
  const std::vector<std::uint8_t> chroma(1, 128);
  const Yuv420View valid{2, 2, plane(y, 2, 1),
                         plane(chroma, 1, 1), plane(chroma, 1, 1)};

  expect_throws<std::invalid_argument>([&]() {
    auto image = valid;
    image.y.row_stride = 0;
    (void)rppg_qnn::yuv420_to_bgr(image);
  });
  expect_throws<std::invalid_argument>([&]() {
    auto image = valid;
    image.u.row_stride = -1;
    (void)rppg_qnn::yuv420_to_bgr(image);
  });
  expect_throws<std::invalid_argument>([&]() {
    auto image = valid;
    image.y.pixel_stride = 0;
    (void)rppg_qnn::yuv420_to_bgr(image);
  });
  expect_throws<std::invalid_argument>([&]() {
    auto image = valid;
    image.v.pixel_stride = 3;
    (void)rppg_qnn::yuv420_to_bgr(image);
  });
  expect_throws<std::invalid_argument>([&]() {
    auto image = valid;
    image.width = 0;
    (void)rppg_qnn::yuv420_to_bgr(image);
  });
  expect_throws<std::invalid_argument>([&]() {
    auto image = valid;
    image.height = -1;
    (void)rppg_qnn::yuv420_to_bgr(image);
  });
}

void rejects_null_planes_and_impossible_large_layouts() {
  const std::vector<std::uint8_t> y(1, 128);
  const YuvPlaneView one = plane(y, 1, 1);

  expect_throws<std::invalid_argument>([&]() {
    const Yuv420View image{1, 1, {nullptr, 1, 1, 1}, one, one};
    (void)rppg_qnn::yuv420_to_bgr(image);
  });

  const auto maximum = std::numeric_limits<int>::max();
  if (static_cast<std::size_t>(maximum) >
      std::numeric_limits<std::size_t>::max() /
          static_cast<std::size_t>(maximum) / 3U) {
    expect_throws<std::overflow_error>([&]() {
      const Yuv420View image{maximum, maximum, one, one, one};
      (void)rppg_qnn::yuv420_to_bgr(image);
    });
  } else {
    expect_throws<std::invalid_argument>([&]() {
      const Yuv420View image{maximum, maximum, one, one, one};
      (void)rppg_qnn::yuv420_to_bgr(image);
    });
  }
}

}  // namespace

int main() {
  converts_contiguous_planar_4x2_to_exact_bgr();
  ignores_padding_at_the_end_of_plane_rows();
  supports_pixel_stride_two_interleaved_chroma_views();
  converts_odd_3x3_dimensions_using_ceil_chroma_extent();
  rejects_truncated_planes();
  rejects_invalid_strides_and_dimensions();
  rejects_null_planes_and_impossible_large_layouts();
  return test_support::finish();
}

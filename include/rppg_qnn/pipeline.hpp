#pragma once

#include "rppg_qnn/config.hpp"
#include "rppg_qnn/deep_runtime.hpp"
#include "rppg_qnn/frame_source.hpp"
#include "rppg_qnn/roi_processor.hpp"
#include "rppg_qnn/result_sink.hpp"

#include <functional>
#include <memory>

namespace rppg_qnn {

struct PipelineDependencies {
  std::function<std::unique_ptr<FrameSource>()> make_source;
  std::function<std::unique_ptr<IRoiProcessor>()> make_roi;
  std::function<std::unique_ptr<IDeepRuntime>()> make_deep_runtime;
  std::function<std::unique_ptr<IResultSink>(const std::filesystem::path&)> make_sink;
};

class Pipeline {
 public:
  Pipeline(AppConfig config, PipelineDependencies dependencies);

  int run();

 private:
  AppConfig config_;
  PipelineDependencies dependencies_;
};

}  // namespace rppg_qnn

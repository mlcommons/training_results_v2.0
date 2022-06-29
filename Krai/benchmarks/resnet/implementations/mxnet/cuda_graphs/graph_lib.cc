#include <mxnet/c_api.h>
#include <mxnet/base.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <unordered_map>
#include <cuda_runtime.h>
#include <iostream>

using namespace mxnet;

#define ERROR(x) \
  std::cerr << x << std::endl;

static std::unordered_map<int, std::unordered_map<int, cudaGraphExec_t>> graph_map;

struct ParamPack {
  int i;
  int rank;

  ParamPack(const int i, const int rank) : i(i), rank(rank) {}
};

void capture(cudaStream_t stream) {
  MSHADOW_CUDA_CALL(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
}

extern "C"
void start_capture(const int i, const int rank, void** inputs, const size_t num_inputs) {
  if (graph_map[rank].count(i) != 0) {
    ERROR("Graph already exists!");
  }
  std::vector<Engine::VarHandle> vars;
  for (size_t i = 0; i < num_inputs; ++i) {
    vars.push_back(reinterpret_cast<NDArray*>(inputs[i])->var());
  }
  Engine::Get()->PushSync([i, rank](RunContext rctx) {
        capture(mshadow::Stream<gpu>::GetStream(rctx.get_stream<mshadow::gpu>()));
      }, Context::GPU(rank), {}, vars,
      FnProperty::kNormal, 0, "Capture");
}

extern "C"
void end_capture(const int i, const int rank, void** outputs, const size_t num_outputs) {
  if (graph_map[rank].count(i) != 0) {
    ERROR("Graph already exists!");
  }
  std::vector<Engine::VarHandle> vars;
  for (size_t i = 0; i < num_outputs; ++i) {
    vars.push_back(reinterpret_cast<NDArray*>(outputs[i])->var());
  }
  Engine::Get()->PushSync([i, rank](RunContext rctx) {
        cudaGraphExec_t* exec = &(graph_map[rank][i]);
        cudaGraph_t g;
        MSHADOW_CUDA_CALL(cudaStreamEndCapture(
              mshadow::Stream<gpu>::GetStream(rctx.get_stream<mshadow::gpu>()), &g));
        cudaGraphNode_t node;
        char buffer[1000];
        MSHADOW_CUDA_CALL(cudaGraphInstantiate(exec, g, &node, buffer, 1000));
      }, Context::GPU(rank), {}, vars,
      FnProperty::kNormal, 0, "EndCapture");
}

extern "C"
void graph_replay(const int i, const int rank, void** inputs, const size_t num_inputs,
                  void** outputs, const size_t num_outputs) {
  if (graph_map[rank].count(i) == 0) {
    ERROR("Graph does not exist!");
  }
  std::vector<Engine::VarHandle> const_vars;
  for (size_t i = 0; i < num_inputs; ++i) {
    const_vars.push_back(reinterpret_cast<NDArray*>(inputs[i])->var());
  }
  std::vector<Engine::VarHandle> vars;
  for (size_t i = 0; i < num_outputs; ++i) {
    vars.push_back(reinterpret_cast<NDArray*>(outputs[i])->var());
  }
  Engine::Get()->PushSync([i, rank](RunContext rctx) {
        cudaGraphExec_t exec = graph_map[rank][i];
        MSHADOW_CUDA_CALL(cudaGraphLaunch(exec,
              mshadow::Stream<gpu>::GetStream(rctx.get_stream<mshadow::gpu>())));
      }, Context::GPU(rank), const_vars, vars,
      FnProperty::kNormal, 0, "Launch");
}

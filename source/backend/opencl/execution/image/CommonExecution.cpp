//
//  CommonExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/CommonExecution.hpp"
#include <chrono>
namespace MNN {
namespace OpenCL {

CommonExecution::CommonExecution(Backend *backend, const MNN::Op *Op)
    : Execution(backend), mOp(Op) {
    mOpType = Op->type();
}

ErrorCode CommonExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime = openCLBackend->getOpenCLRuntime();
    MNN::ScopedTrace trace(std::string("MNN/OCL/OpResize/") + EnumNameOpType(mOpType));
    openCLBackend->startRecord(mRecording);
    
    auto error = onEncode(inputs, outputs);
    if(NO_ERROR != error){
        return error;
    }
    
    for (auto &unit : mUnits) {
        bool lws_null = true;
        for (size_t i = 0; i < unit.globalWorkSize.dimensions(); ++i) {
            unit.globalWorkSize.get()[i] = ROUND_UP(unit.globalWorkSize.get()[i], std::max((size_t)1, unit.localWorkSize.get()[i]));
            if(unit.localWorkSize.get()[i] != 0) {
                lws_null = false;
            }
        }
        if(lws_null){
            unit.localWorkSize = cl::NullRange;
        }
    }
    openCLBackend->endRecord(mRecording);
    return NO_ERROR;
}

ErrorCode CommonExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());
    auto runtime = openCLBackend->getOpenCLRuntime();
    MNN::ScopedTrace trace(std::string("MNN/OCL/OpExecute/") + EnumNameOpType(mOpType));
    int idx = 0;
#ifndef ENABLE_OPENCL_TIME_PROFILER
    if(openCLBackend->isUseRecordQueue()){
        openCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
        return NO_ERROR;
    }
#endif
    auto res = CL_SUCCESS;
    for (auto &unit : mUnits) {
        auto enqueueBegin = std::chrono::steady_clock::now();
        auto kernelName = std::string(EnumNameOpType(mOpType)) + "#" + std::to_string(idx);
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                    cl::NullRange,
                                                    unit.globalWorkSize,
                                                    unit.localWorkSize,
                                                    nullptr,
                                                    &event);
        auto enqueueCostUs = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - enqueueBegin).count();
        std::vector<uint32_t> gws;
        std::vector<uint32_t> lws;
        for (uint32_t i = 0; i < unit.globalWorkSize.dimensions(); ++i) {
            gws.emplace_back((uint32_t)unit.globalWorkSize.get()[i]);
        }
        if (unit.localWorkSize.dimensions() > 0) {
            for (uint32_t i = 0; i < unit.localWorkSize.dimensions(); ++i) {
                lws.emplace_back((uint32_t)unit.localWorkSize.get()[i]);
            }
        }
        runtime->pushEvent(kernelName, event, enqueueCostUs, gws, lws);
    #else
        res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                    cl::NullRange,
                                                    unit.globalWorkSize,
                                                    unit.localWorkSize);
        auto enqueueCostUs = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - enqueueBegin).count();
        if (runtime->isLogEnabled()) {
            runtime->logProfile("KernelEnqueue", "name=" + kernelName + ", host_enqueue_us=" + std::to_string(enqueueCostUs));
        }
    #endif
        idx++;
        MNN_CHECK_CL_SUCCESS(res, EnumNameOpType(mOp->type()));
    }
    return NO_ERROR;
}
} // namespace OpenCL
}; // namespace MNN

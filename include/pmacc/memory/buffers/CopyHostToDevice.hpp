
#pragma once

#include <pmacc/types.hpp>
#include <pmacc/memory/buffers/CopyTask.hpp>
#include <pmacc/tasks/StreamTask.hpp>
#include <rmngr/task.hpp>

#include "WaitForDevice.hpp"

namespace pmacc
{

template <class T, unsigned T_DIM>
class HostBuffer;
template <class T, unsigned T_DIM>
class DeviceBuffer;

namespace memory
{
namespace buffers
{

struct LabelHostToDevice
{
    void properties(Scheduler::Schedulable& s)
    {
        s.proto_property< GraphvizPolicy >().label = "CopyHostToDevice()";
    }
};

template <typename Impl, typename T, unsigned T_DIM>
class TaskCopyHostToDeviceBase
    : public rmngr::Task<
          Impl,
          boost::mpl::vector<
	      NEW::StreamTask,
	      CopyTask<
	          HostBuffer<T, T_DIM>,
	          DeviceBuffer<T, T_DIM>
              >,
              LabelHostToDevice
          >
      >
{
public:
    TaskCopyHostToDeviceBase(HostBuffer<T, T_DIM>& src, DeviceBuffer<T, T_DIM>& dst)
    {
        this->src =  &src;
	this->dst =  &dst;
    }

    void run()
    {
        size_t current_size = this->src->getCurrentSize();
	DataSpace<T_DIM> hostCurrentSize = this->src->getCurrentDataSpace(current_size);

        this->dst->setCurrentSize(current_size);
        if (this->src->is1D() && this->dst->is1D())
            fastCopy(this->src->getPointer(), this->dst->getPointer(), hostCurrentSize.productOfComponents());
        else
	    copy(hostCurrentSize);

        TaskWaitForDevice::create( Scheduler::getInstance() );
    }

protected:    
    virtual void copy(DataSpace<T_DIM> &hostCurrentSize) = 0;

    void fastCopy(T* src, T* dst, size_t size)
    {
        CUDA_CHECK(cudaMemcpyAsync(dst,
				   src,
				   size * sizeof (T),
				   cudaMemcpyHostToDevice,
				   this->getCudaStream()));
    }
};

template <class T, unsigned T_DIM>
class TaskCopyHostToDevice;

template <class T>
class TaskCopyHostToDevice<T, DIM1>
    : public TaskCopyHostToDeviceBase<
          TaskCopyHostToDevice<T, DIM1>,
          T,
          DIM1
      >
{
public:
    TaskCopyHostToDevice(HostBuffer<T, DIM1>& src, DeviceBuffer<T, DIM1>& dst)
        : TaskCopyHostToDeviceBase<TaskCopyHostToDevice<T, DIM1>, T, DIM1>(src, dst)
    {}

private:
    void copy(DataSpace<DIM1> & hostCurrentSize)
    {
        CUDA_CHECK(cudaMemcpyAsync(this->dst->getPointer(), /*pointer include X offset*/
				   this->src->getBasePointer(),
				   hostCurrentSize[0] * sizeof (T), cudaMemcpyHostToDevice,
				   this->getCudaStream()));
    }
};

template <class T>
class TaskCopyHostToDevice<T, DIM2>
    : public TaskCopyHostToDeviceBase<
          TaskCopyHostToDevice<T, DIM2>,
          T,
          DIM2
      >
{
public:
    TaskCopyHostToDevice( HostBuffer<T, DIM2>& src, DeviceBuffer<T, DIM2>& dst)
      : TaskCopyHostToDeviceBase<TaskCopyHostToDevice<T, DIM2>, T, DIM2>(src, dst)
    {}

private:
    void copy(DataSpace<DIM2> &hostCurrentSize)
    {
        CUDA_CHECK(cudaMemcpy2DAsync(this->dst->getPointer(),
				     this->dst->getPitch(), /*this is pitch*/
				     this->src->getBasePointer(),
				     this->src->getDataSpace()[0] * sizeof (T), /*this is pitch*/
				     hostCurrentSize[0] * sizeof (T),
				     hostCurrentSize[1],
				     cudaMemcpyHostToDevice,
				     this->getCudaStream()));
    }
};

template <class T>
class TaskCopyHostToDevice<T, DIM3>
    : public TaskCopyHostToDeviceBase<
          TaskCopyHostToDevice<T, DIM3>,
          T,
          DIM3
      >
{
public:
    TaskCopyHostToDevice( HostBuffer<T, DIM3>& src, DeviceBuffer<T, DIM3>& dst)
        : TaskCopyHostToDeviceBase<TaskCopyHostToDevice<T, DIM3>, T, DIM3>(src, dst)
    {}

private:
    void copy(DataSpace<DIM3> &hostCurrentSize)
    {
        cudaPitchedPtr hostPtr;
	hostPtr.pitch = this->src->getDataSpace()[0] * sizeof (T);
	hostPtr.ptr = this->src->getBasePointer();
	hostPtr.xsize = this->src->getDataSpace()[0] * sizeof (T);
	hostPtr.ysize = this->src->getDataSpace()[1];

	cudaMemcpy3DParms params;
	params.dstArray = nullptr;
	params.dstPos = make_cudaPos(this->dst->getOffset()[0] * sizeof (T),
				     this->dst->getOffset()[1],
				     this->dst->getOffset()[2]);
	params.dstPtr = this->dst->getCudaPitched();

	params.srcArray = nullptr;
	params.srcPos = make_cudaPos(0, 0, 0);
	params.srcPtr = hostPtr;

	params.extent = make_cudaExtent(
		            hostCurrentSize[0] * sizeof (T),
			    hostCurrentSize[1],
			    hostCurrentSize[2]);
	params.kind = cudaMemcpyHostToDevice;

	CUDA_CHECK(cudaMemcpy3DAsync(&params, this->getCudaStream()));
    }
};

} // namespace buffers

} // namespace memory

} // namespace pmacc

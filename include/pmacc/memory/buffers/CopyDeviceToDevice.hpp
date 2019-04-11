#include <pmacc/types.hpp>
#include <pmacc/memory/buffers/CopyTask.hpp>
#include <pmacc/tasks/StreamTask.hpp>
#include <rmngr/task.hpp>

#include "WaitForDevice.hpp"

namespace pmacc
{

template < typename T, unsigned T_Dim >
class HostBuffer;

template < typename T, unsigned T_Dim >
class DeviceBuffer;

namespace memory
{
namespace buffers
{

struct LabelDeviceToDevice
{
    void properties(Scheduler::Schedulable& s)
    {
        s.proto_property< GraphvizPolicy >().label = "CopyDeviceToDevice()";
    }
};

template <
    typename Impl,
    typename T,
    size_t T_Dim
>
class TaskCopyDeviceToDeviceBase
    : public rmngr::Task<
          Impl,
          boost::mpl::vector<
	    NEW::StreamTask,
              CopyTask<
                  DeviceBuffer<T, T_Dim>,
                  DeviceBuffer<T, T_Dim>
              >,
              LabelDeviceToDevice
          >
      >
{
public:
    TaskCopyDeviceToDeviceBase(
        DeviceBuffer<T, T_Dim> & src,
        DeviceBuffer<T, T_Dim> & dst
    )
    {
        this->src = &src;
        this->dst = &dst;
    }

    void run()
    {
        size_t current_size = this->src->getCurrentSize();
	this->dst->setCurrentSize(current_size);
	DataSpace<T_Dim> devCurrentSize = this->src->getCurrentDataSpace(current_size);
	if (this->src->is1D() && this->dst->is1D())
	    fastCopy(this->src->getPointer(), this->dst->getPointer(), devCurrentSize.productOfComponents());
	else
	    copy(devCurrentSize);
    }

protected:    
    virtual void copy(DataSpace<T_Dim> &devCurrentSize) = 0;

    void fastCopy(T* src, T* dst, size_t size)
    {
        CUDA_CHECK(cudaMemcpyAsync(dst,
				   src,
				   size * sizeof (T),
				   cudaMemcpyDeviceToDevice,
				   this->getCudaStream()));
    }
};

template <class T, unsigned T_DIM>
class TaskCopyDeviceToDevice;

template <class T>
class TaskCopyDeviceToDevice<T, DIM1>
    : public TaskCopyDeviceToDeviceBase<
          TaskCopyDeviceToDevice<T, DIM1>,
          T,
          DIM1
      >
{
public:
    TaskCopyDeviceToDevice(DeviceBuffer<T, DIM1>& src, DeviceBuffer<T, DIM1>& dst)
        : TaskCopyDeviceToDeviceBase<TaskCopyDeviceToDevice<T, DIM1>, T, DIM1>(src, dst)
    {}

private:
    void copy(DataSpace<DIM1> &devCurrentSize)
    {
        CUDA_CHECK(cudaMemcpyAsync(this->dst->getPointer(),
				   this->src->getPointer(),
				   devCurrentSize[0] * sizeof (T),
				   cudaMemcpyDeviceToDevice,
				   this->getCudaStream()));
    }
};

template <class T>
class TaskCopyDeviceToDevice<T, DIM2>
    : public TaskCopyDeviceToDeviceBase<
          TaskCopyDeviceToDevice<T, DIM2>,
          T,
          DIM2
      >
{
public:
    TaskCopyDeviceToDevice( DeviceBuffer<T, DIM2>& src, DeviceBuffer<T, DIM2>& dst )
        : TaskCopyDeviceToDeviceBase<TaskCopyDeviceToDevice<T, DIM2>, T, DIM2>(src, dst)
    {}

private:
    void copy(DataSpace<DIM2> & devCurrentSize)
    {
        CUDA_CHECK(cudaMemcpy2DAsync(this->dst->getPointer(),
				     this->dst->getPitch(),
				     this->src->getPointer(),
				     this->src->getPitch(),
				     devCurrentSize[0] * sizeof (T),
				     devCurrentSize[1],
				     cudaMemcpyDeviceToDevice,
				     this->getCudaStream()));

    }
};

template <class T>
class TaskCopyDeviceToDevice<T, DIM3>
    : public TaskCopyDeviceToDeviceBase<
          TaskCopyDeviceToDevice<T, DIM3>,
          T,
          DIM3
      >
{
public:
    TaskCopyDeviceToDevice( DeviceBuffer<T, DIM3> & src, DeviceBuffer<T, DIM3> & dst )
      : TaskCopyDeviceToDeviceBase<TaskCopyDeviceToDevice<T, DIM3>, T , DIM3>(src, dst)
    {}

private:
    void copy(DataSpace<DIM3> & devCurrentSize)
    {
        cudaMemcpy3DParms params;
	params.srcArray = nullptr;
	params.srcPos = make_cudaPos(
		            this->src->getOffset()[0] * sizeof (T),
			    this->src->getOffset()[1],
			    this->src->getOffset()[2]);
	params.srcPtr = this->src->getCudaPitched();

	params.dstArray = nullptr;
	params.dstPos = make_cudaPos(
			    this->dst->getOffset()[0] * sizeof (T),
			    this->dst->getOffset()[1],
			    this->dst->getOffset()[2]);
	params.dstPtr = this->dst->getCudaPitched();

	params.extent = make_cudaExtent(
		            devCurrentSize[0] * sizeof (T),
			    devCurrentSize[1],
			    devCurrentSize[2]);

	params.kind = cudaMemcpyDeviceToDevice;
	CUDA_CHECK(cudaMemcpy3DAsync(&params, this->getCudaStream()));
    }
};

} // namespace buffers

} // namespace memory

} // namespace pmacc


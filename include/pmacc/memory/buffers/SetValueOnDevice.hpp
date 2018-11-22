
#pragma once

#include <pmacc/tasks/StreamTask.hpp>


namespace pmacc
{

struct KernelSetValueOnDeviceMemory
{
    template< typename T_Acc >
    DINLINE void operator()(const T_Acc&, size_t* pointer, const size_t size) const
    {
        *pointer = size;
    }
};
  /*
template < typename T, size_t T_Dim >
class TaskSetValueOnDevice
  : public StreamTask
{
    TaskSetValueOnDevice( DeviceBuffer<T, T_Dim> & dst, size_t s )
      : destination(dst), size(s)
    {}

    void run()
    {
        auto sizePtr = destination.getCurrentSizeOnDevicePointer();
        CUPLA_KERNEL( KernelSetValueOnDeviceMemory )(
            1,
            1,
            0,
            stream_task.getCudaStream()
        )(
            sizePtr,
            size
        );
    }

    DeviceBuffer<T, T_Dim> & destination;
    size_t const size;
};
*/
} // namespace pmacc


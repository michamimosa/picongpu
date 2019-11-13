#pragma once

#include <pmacc/types.hpp>
#include <pmacc/memory/buffers/WaitForDevice.hpp>

namespace pmacc
{

template < typename T, std::size_t T_Dim >
class DeviceBuffer;

namespace memory
{
namespace buffers
{

namespace device2device_detail
{

template < typename T >
void fast_copy(
    T * dst,
    T * src,
    size_t size
)
{
    cudaStream_t cuda_stream = 0;
    CUDA_CHECK(cudaMemcpyAsync(dst,
                               src,
                               size * sizeof (T),
                               cudaMemcpyDeviceToDevice,
                               cuda_stream));
}

template < typename T >
void copy(
    DeviceBuffer<T, DIM1> & dst,
    DeviceBuffer<T, DIM1> & src,
    DataSpace<DIM1> & size
)
{
    cudaStream_t cuda_stream = 0;
    CUDA_CHECK(cudaMemcpyAsync(dst.getPointer(),
                               src.getPointer(),
                               size[0] * sizeof (T),
                               cudaMemcpyDeviceToDevice,
                               cuda_stream));
}

template < typename T >
void copy(
    DeviceBuffer<T, DIM2> & dst,
    DeviceBuffer<T, DIM2> & src,
    DataSpace<DIM2> & size
)
{
    cudaStream_t cuda_stream = 0;
    CUDA_CHECK(cudaMemcpy2DAsync(dst.getPointer(),
                                 dst.getPitch(),
                                 src.getPointer(),
                                 src.getPitch(),
                                 size[0] * sizeof (T),
                                 size[1],
                                 cudaMemcpyDeviceToDevice,
                                 cuda_stream));

}

template < typename T >
void copy(
    DeviceBuffer<T, DIM3> & dst,
    DeviceBuffer<T, DIM3> & src,
    DataSpace<DIM3> & size
)
{
    cudaStream_t cuda_stream = 0;

    cudaMemcpy3DParms params;
    params.srcArray = nullptr;
    params.srcPos = make_cudaPos(src.getOffset()[0] * sizeof (T),
                                 src.getOffset()[1],
                                 src.getOffset()[2]);
    params.srcPtr = src.getCudaPitched();

    params.dstArray = nullptr;
    params.dstPos = make_cudaPos(dst.getOffset()[0] * sizeof (T),
                                 dst.getOffset()[1],
                                 dst.getOffset()[2]);
    params.dstPtr = dst.getCudaPitched();

    params.extent = make_cudaExtent(size[0] * sizeof (T),
                                    size[1],
                                    size[2]);

    params.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy3DAsync(&params, cuda_stream));
}

} // namespace device2device_detail

template <
    typename T,
    std::size_t T_Dim
>
void
copy(
    DeviceBuffer<T, T_Dim> & dst,
    DeviceBuffer<T, T_Dim> & src
)
{
    Environment<>::get().ResourceManager().emplace_task(
        [&dst, &src]
        {
            size_t current_size = src.getCurrentSize();
            dst.setCurrentSize(current_size);

            DataSpace<T_Dim> devCurrentSize = src.getCurrentDataSpace(current_size);
            if (src.is1D() && dst.is1D())
                device2device_detail::fast_copy(dst.getPointer(), src.getPointer(), devCurrentSize.productOfComponents());
            else
                device2device_detail::copy(dst, src, devCurrentSize);
        },
        TaskProperties::Builder()
            .label("copyDeviceToDevice")
            .resources({
                dst.write(),
                dst.size_resource.write(),
                src.write(),
                src.size_resource.write(),
                cuda_resources::streams[0].write()
            })
    );
}

} // namespace buffers

} // namespace memory

} // namespace pmacc



#pragma once

#include <pmacc/types.hpp>

namespace pmacc
{

template <class T, std::size_t T_Dim>
class HostBuffer;

template <class T, std::size_t T_Dim>
class DeviceBuffer;

namespace memory
{
namespace buffers
{

namespace host2device_detail
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
                               cudaMemcpyHostToDevice,
                               cuda_stream));
}

template < typename T >
void copy(
    DeviceBuffer<T, DIM1> & dst,
    HostBuffer<T, DIM1> & src,
    DataSpace<DIM1> & size
)
{
    cudaStream_t cuda_stream = 0;
    CUDA_CHECK(cudaMemcpyAsync(dst.getPointer(), /*pointer include X offset*/
                               src.getBasePointer(),
                               size[0] * sizeof (T),
                               cudaMemcpyHostToDevice,
                               cuda_stream));
}

template < typename T >
void copy(
    DeviceBuffer<T, DIM2> & dst,
    HostBuffer<T, DIM2> & src,
    DataSpace<DIM2> & size
)
{
    cudaStream_t cuda_stream = 0;
    CUDA_CHECK(cudaMemcpy2DAsync(dst.getPointer(),
                                 dst.getPitch(), /*this is pitch*/
                                 src.getBasePointer(),
                                 src.getDataSpace()[0] * sizeof (T), /*this is pitch*/
                                 size[0] * sizeof (T),
                                 size[1],
                                 cudaMemcpyHostToDevice,
                                 cuda_stream));

}

template < typename T >
void copy(
    DeviceBuffer<T, DIM3> & dst,
    HostBuffer<T, DIM3> & src,
    DataSpace<DIM3> & size
)
{
    cudaStream_t cuda_stream = 0;

    cudaPitchedPtr hostPtr;
    hostPtr.pitch = src.getDataSpace()[0] * sizeof (T);
    hostPtr.ptr = src.getBasePointer();
    hostPtr.xsize = src.getDataSpace()[0] * sizeof (T);
    hostPtr.ysize = src.getDataSpace()[1];

    cudaMemcpy3DParms params;
    params.dstArray = nullptr;
    params.dstPos = make_cudaPos(dst.getOffset()[0] * sizeof (T),
                                 dst.getOffset()[1],
                                 dst.getOffset()[2]);
    params.dstPtr = dst.getCudaPitched();

    params.srcArray = nullptr;
    params.srcPos = make_cudaPos(0, 0, 0);
    params.srcPtr = hostPtr;

    params.extent = make_cudaExtent(
                                    size[0] * sizeof (T),
                                    size[1],
                                    size[2]);
    params.kind = cudaMemcpyHostToDevice;

    CUDA_CHECK(cudaMemcpy3DAsync(&params, cuda_stream));
}

} // namespace host2device_detail



template <
    typename T,
    std::size_t T_Dim
>
void
copy(
    DeviceBuffer<T, T_Dim> & dst,
    HostBuffer<T, T_Dim> & src
)
{
    Scheduler::Properties prop;
    prop.policy< rmngr::ResourceUserPolicy >() += cuda_resources::streams[0].write();
    prop.policy< rmngr::ResourceUserPolicy >() += dst.write();
    prop.policy< rmngr::ResourceUserPolicy >() += dst.size_resource.write();
    prop.policy< rmngr::ResourceUserPolicy >() += src.read();
    prop.policy< rmngr::ResourceUserPolicy >() += src.size_resource.write();
    prop.policy< GraphvizPolicy >().label = "copyHostToDevice";

    Scheduler::emplace_task(
        [&dst, &src]
        {
            size_t current_size = src.getCurrentSize();

            dst.setCurrentSize(current_size);
            DataSpace<T_Dim> devCurrentSize = src.getCurrentDataSpace(current_size);

            if (src.is1D() && dst.is1D())
                host2device_detail::fast_copy(dst.getPointer(),
                                              src.getPointer(),
                                              devCurrentSize.productOfComponents());
            else
                host2device_detail::copy(dst, src, devCurrentSize);

            task_synchronize_stream(0);
        },
        prop
    );
}

} // namespace buffers

} // namespace memory

} // namespace pmacc

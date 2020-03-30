#pragma once

#include <pmacc/types.hpp>
#include <pmacc/memory/buffers_new/DeviceBuffer.hpp>

namespace pmacc
{
namespace mem
{
namespace buffer
{

namespace device2device_detail
{

template < typename T >
void fast_copy(
    T * dst,
    T const * src,
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

template <
    typename T,
    typename T_DstDataAccessPolicy,
    typename T_SrcDataAccessPolicy
>
void copy(
    device_buffer::data::WriteGuard< T, DIM1, T_DstDataAccessPolicy > const & dst,
    device_buffer::data::ReadGuard< T, DIM1, T_SrcDataAccessPolicy > const & src,
    DataSpace<DIM1> const & size
)
{
    cudaStream_t cuda_stream = 0;
    CUDA_CHECK(cudaMemcpyAsync(dst.getPointer(),
                               src.getPointer(),
                               size[0] * sizeof (T),
                               cudaMemcpyDeviceToDevice,
                               cuda_stream));
}

template <
    typename T,
    typename T_DstDataAccessPolicy,
    typename T_SrcDataAccessPolicy
>
void copy(
    device_buffer::data::WriteGuard< T, DIM2, T_DstDataAccessPolicy > const & dst,
    device_buffer::data::ReadGuard< T, DIM2, T_SrcDataAccessPolicy > const & src,
    DataSpace<DIM2> const & size
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

template <
    typename T,
    typename T_DstDataAccessPolicy,
    typename T_SrcDataAccessPolicy
>
void copy(
    device_buffer::data::WriteGuard< T, DIM3, T_DstDataAccessPolicy > const & dst,
    device_buffer::data::ReadGuard< T, DIM3, T_SrcDataAccessPolicy > const & src,
    DataSpace<DIM3> const & size
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
    std::size_t T_Dim,
    typename T_DstDataAccessPolicy,
    typename T_SrcDataAccessPolicy
>
void
copy(
    device_buffer::WriteGuard< T, T_Dim, T_DstDataAccessPolicy > const & dst,
    device_buffer::ReadGuard< T, T_Dim, T_SrcDataAccessPolicy > const & src
)
{
    Environment<>::task(
        []( auto dst, auto src, auto cuda_stream )
        {
            dst.size().set( src.size().get() );

            DataSpace<T_Dim> devCurrentSize = src.size().getCurrentDataSpace();
            if (src.data().is1D() && dst.data().is1D())
                device2device_detail::fast_copy(dst.data().getPointer(), src.data().getPointer(), devCurrentSize.productOfComponents());
            else
                device2device_detail::copy(dst.data(), src.data(), devCurrentSize);
        },
        TaskProperties::Builder()
            .label("pmacc::mem::copy(dst: Device, src: Device)"),
        dst.write(),
        src.read(),
        Environment<>::get().cuda_stream()
    );
}

} // namespace buffer

} // namespace mem

} // namespace pmacc


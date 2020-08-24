
#pragma once

#include <pmacc/types.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/memory/buffers/HostBuffer.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>

namespace pmacc
{
namespace mem
{
namespace buffer
{

namespace host2device_detail
{
    
template < typename T >
void fast_copy(
    T * dst,
    T const * src,
    size_t size
)
{
    cuplaStream_t cuda_stream = 0;
    CUDA_CHECK(cuplaMemcpyAsync(dst,
                               src,
                               size * sizeof (T),
                               cuplaMemcpyHostToDevice,
                               cuda_stream));
}

template <
    typename T,
    typename T_DstDataAccessPolicy,
    typename T_SrcDataAccessPolicy
>
void copy(
    device_buffer::data::WriteGuard< T, DIM1, T_DstDataAccessPolicy > const & dst,
    host_buffer::data::ReadGuard< T, DIM1, T_SrcDataAccessPolicy > const & src,
    DataSpace<DIM1> const & size
)
{
    cuplaStream_t cuda_stream = 0;
    CUDA_CHECK(cuplaMemcpyAsync(dst.getPointer(), /*pointer include X offset*/
                               src.getPointer(),
                               size[0] * sizeof (T),
                               cuplaMemcpyHostToDevice,
                               cuda_stream));
}

template <
    typename T,
    typename T_DstDataAccessPolicy,
    typename T_SrcDataAccessPolicy
>
void copy(
    device_buffer::data::WriteGuard< T, DIM2, T_SrcDataAccessPolicy > const & dst,
    host_buffer::data::ReadGuard< T, DIM2, T_DstDataAccessPolicy > const & src,
    DataSpace<DIM2> const & size
)
{
    cuplaStream_t cuda_stream = 0;
    CUDA_CHECK(cuplaMemcpy2DAsync(
        dst.getPointer(),
        dst.getPitch(),
        src.getPointer(),
        src.getPitch(),
        size[0] * sizeof (T),
        size[1],
        cuplaMemcpyHostToDevice,
        cuda_stream));

}

template <
    typename T,
    typename T_DstDataAccessPolicy,
    typename T_SrcDataAccessPolicy
>
void copy(
    device_buffer::data::WriteGuard< T, DIM3, T_SrcDataAccessPolicy > const & dst,
    host_buffer::data::ReadGuard< T, DIM3, T_DstDataAccessPolicy > const & src,
    DataSpace<DIM3> const & size
)
{
    cuplaStream_t cuda_stream = 0;
    cuplaPitchedPtr hostPtr;
    hostPtr.ptr = src.getBasePointer();
    hostPtr.pitch = src.getPitch();
    hostPtr.xsize = src.getDataSpace()[0] * sizeof (T);
    hostPtr.ysize = src.getDataSpace()[1];

    cuplaMemcpy3DParms params;
    params.dstArray = nullptr;
    params.dstPos = make_cuplaPos(
       dst.getOffset()[0] * sizeof (T),
       dst.getOffset()[1],
       dst.getOffset()[2]);
    params.dstPtr = dst.getCudaPitched();

    params.srcArray = nullptr;
    params.srcPos = make_cuplaPos(
        src.getOffset()[0] * sizeof(T),
        src.getOffset()[1],
        src.getOffset()[2]);
    params.srcPtr.ptr = hostPtr;

    params.extent = make_cuplaExtent(
        size[0] * sizeof (T),
        size[1],
        size[2]);
    params.kind = cuplaMemcpyHostToDevice;

    CUDA_CHECK(cuplaMemcpy3DAsync(&params, cuda_stream));
}

} // namespace host2device_detail

template <
    typename T,
    std::size_t T_Dim,
    typename T_DstDataAccessPolicy,
    typename T_SrcDataAccessPolicy
>
void
copy(
     device_buffer::WriteGuard< T, T_Dim, T_DstDataAccessPolicy > const & dst,
     host_buffer::ReadGuard< T, T_Dim, T_SrcDataAccessPolicy > const & src
)
{
    Environment<>::task(
        []( auto dst, auto src, auto cuda_stream )
        {
            dst.size().set( src.size().get() );

            DataSpace<T_Dim> devCurrentSize = src.size().getCurrentDataSpace();

            if (src.data().is1D() && dst.data().is1D())
                host2device_detail::fast_copy(dst.data().getPointer(),
                                              src.data().getPointer(),
                                              devCurrentSize.productOfComponents());
            else
                host2device_detail::copy(dst.data(), src.data(), devCurrentSize);

            cuda_stream.sync();
        },
        TaskProperties::Builder().label("pmacc::mem::copy(dst: Device, src: Host)"),
        dst.write(),
        src.read(),
        Environment<>::get().cuda_stream()
    );
}

} // namespace buffer

} // namespace mem

} // namespace pmacc


#pragma once

#include <pmacc/types.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>

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
    CUDA_CHECK(cuplaMemcpyAsync(dst,
                               src,
                               size * sizeof (T),
                               cuplaMemcpyDeviceToDevice,
                               redGrapes::thread::current_cupla_stream));
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
    CUDA_CHECK(cuplaMemcpyAsync(dst.getPointer(),
                               src.getPointer(),
                               size[0] * sizeof (T),
                               cuplaMemcpyDeviceToDevice,
                               redGrapes::thread::current_cupla_stream));
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
    CUDA_CHECK(cuplaMemcpy2DAsync(dst.getPointer(),
                                 dst.getPitch(),
                                 src.getPointer(),
                                 src.getPitch(),
                                 size[0] * sizeof (T),
                                 size[1],
                                 cuplaMemcpyDeviceToDevice,
                                 redGrapes::thread::current_cupla_stream));

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
    cuplaMemcpy3DParms params;
    params.srcArray = nullptr;
    params.srcPos = make_cuplaPos(src.getOffset()[0] * sizeof (T),
                                 src.getOffset()[1],
                                 src.getOffset()[2]);
    params.srcPtr = src.getCudaPitched();

    params.dstArray = nullptr;
    params.dstPos = make_cuplaPos(dst.getOffset()[0] * sizeof (T),
                                 dst.getOffset()[1],
                                 dst.getOffset()[2]);
    params.dstPtr = dst.getCudaPitched();

    params.extent = make_cuplaExtent(size[0] * sizeof (T),
                                    size[1],
                                    size[2]);

    params.kind = cuplaMemcpyDeviceToDevice;
    CUDA_CHECK(cuplaMemcpy3DAsync(&params, redGrapes::thread::current_cupla_stream));
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
        []( auto dst, auto src )
        {
            dst.size().set( src.size().get() );

            DataSpace< T_Dim > devCurrentSize = src.size().getCurrentDataSpace();

            Environment<>::task(
                [devCurrentSize]( auto dst, auto src )
                {
                    if (src.is1D() && dst.is1D())
                        device2device_detail::fast_copy(
                            dst.getPointer(),
                            src.getPointer(),
                            devCurrentSize.productOfComponents()
                        );
                    else
                        device2device_detail::copy(dst, src, devCurrentSize);                                    
                },
                TaskProperties::Builder()
                    .label("cuplaMemcpyAsync(dst: Device, src: Device)")
                    .scheduling_tags({ SCHED_CUPLA }),
                dst.data(),
                src.data()
            );
        },
        TaskProperties::Builder()
            .label("pmacc::mem::copy(dst: Device, src: Device)"),
        dst.write(),
        src.read()
    );
}

} // namespace buffer

} // namespace mem

} // namespace pmacc


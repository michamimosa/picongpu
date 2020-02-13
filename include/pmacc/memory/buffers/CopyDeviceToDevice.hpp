#pragma once

#include <pmacc/types.hpp>

namespace pmacc
{

template < typename T, std::size_t T_Dim >
class DeviceBuffer;

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

template < typename T >
void copy(
    buffer::data::WriteGuard< DeviceBuffer<T, DIM1> > dst,
    buffer::data::ReadGuard< DeviceBuffer<T, DIM1> > src,
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
    buffer::data::WriteGuard< DeviceBuffer<T, DIM2> > dst,
    buffer::data::ReadGuard< DeviceBuffer<T, DIM2> > src,
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
    buffer::data::WriteGuard< DeviceBuffer<T, DIM3> > dst,
    buffer::data::ReadGuard< DeviceBuffer<T, DIM3> > src,
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
    WriteGuard< DeviceBuffer<T, T_Dim> > dst,
    ReadGuard< DeviceBuffer<T, T_Dim> > src
)
{
    Environment<>::task(
        []( auto dst, auto src, auto cuda_stream )
        {
            size_t current_size = src.size().get();
            dst.size().set(current_size);

            DataSpace<T_Dim> devCurrentSize = src.size().getCurrentDataSpace();
            if (src.data().is1D() && dst.data().is1D())
                device2device_detail::fast_copy(dst.data().getPointer(), src.data().getPointer(), devCurrentSize.productOfComponents());
            else
                device2device_detail::copy(dst.data(), src.data(), devCurrentSize);
        },
        TaskProperties::Builder()
            .label("copyDeviceToDevice"),
        std::move(dst),
        std::move(src),
        Environment<>::get().cuda_stream()
    );
}

} // namespace buffer

} // namespace mem

} // namespace pmacc


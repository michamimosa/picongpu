
#pragma once

#include <pmacc/types.hpp>

namespace pmacc
{

template <class T, std::size_t T_Dim, typename T_DataAccessPolicy>
class HostBuffer;

template <class T, std::size_t T_Dim, typename T_DataAccessPolicy>
class DeviceBuffer;

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
    cudaStream_t cuda_stream = 0;
    CUDA_CHECK(cudaMemcpyAsync(dst,
                               src,
                               size * sizeof (T),
                               cudaMemcpyHostToDevice,
                               cuda_stream));
}

template < typename T, typename T_DataAccessPolicy >
void copy(
    device_buffer::data::WriteGuard< DeviceBuffer<T, DIM1, T_DataAccessPolicy> > dst,
    buffer::data::ReadGuard< HostBuffer<T, DIM1, T_DataAccessPolicy> > src,
    DataSpace<DIM1> const & size
)
{
    cudaStream_t cuda_stream = 0;
    CUDA_CHECK(cudaMemcpyAsync(dst.getPointer(), /*pointer include X offset*/
                               src.getBasePointer(),
                               size[0] * sizeof (T),
                               cudaMemcpyHostToDevice,
                               cuda_stream));
}

template < typename T, typename T_DataAccessPolicy >
void copy(
    device_buffer::data::WriteGuard< DeviceBuffer<T, DIM2, T_DataAccessPolicy> > dst,
    buffer::data::ReadGuard< HostBuffer<T, DIM2, T_DataAccessPolicy> > src,
    DataSpace<DIM2> const & size
)
{
    cudaStream_t cuda_stream = 0;
    CUDA_CHECK(cudaMemcpy2DAsync(dst.getPointer(),
                                 dst.getPitch(), /*this is pitch*/
                                 src.getBasePointer(),
                                 size[0] * sizeof (T), /*this is pitch*/
                                 size[0] * sizeof (T),
                                 size[1],
                                 cudaMemcpyHostToDevice,
                                 cuda_stream));

}

template < typename T, typename T_DataAccessPolicy >
void copy(
    device_buffer::data::WriteGuard< DeviceBuffer<T, DIM3, T_DataAccessPolicy> > dst,
    buffer::data::ReadGuard< HostBuffer<T, DIM3, T_DataAccessPolicy> > src,
    DataSpace<DIM3> const & size
)
{
    cudaStream_t cuda_stream = 0;

    cudaPitchedPtr hostPtr;
    hostPtr.pitch = size[0] * sizeof (T);
    hostPtr.ptr = src.getBasePointer();
    hostPtr.xsize = size[0] * sizeof (T);
    hostPtr.ysize = size[1];

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
    std::size_t T_Dim,
    typename T_DataAccessPolicy
>
void
copy(
     buffer::WriteGuard< DeviceBuffer<T, T_Dim, T_DataAccessPolicy> > dst,
     buffer::ReadGuard< HostBuffer<T, T_Dim, T_DataAccessPolicy> > src
)
{
    Environment<>::task(
        []( auto dst, auto src, auto cuda_stream )
        {
            size_t current_size = src.size().get();
            dst.size().set(current_size);

            DataSpace<T_Dim> devCurrentSize = src.size().data_space();

            if (src.data().is1D() && dst.data().is1D())
                host2device_detail::fast_copy(dst.data().getPointer(),
                                              src.data().getPointer(),
                                              devCurrentSize.productOfComponents());
            else
                host2device_detail::copy(dst.data(), src.data(), devCurrentSize);

            cuda_stream->sync();
        },
        TaskProperties::Builder()
            .label("copyHostToDevice"),
        std::move(dst),
        std::move(src),
        Environment<>::get().cuda_stream()
    );
}

} // namespace buffer

} // namespace mem

} // namespace pmacc

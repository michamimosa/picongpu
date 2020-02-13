/* Copyright 2013-2020 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz, Michael Sippel
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <iomanip>
#include <pmacc/types.hpp>
#include <pmacc/Environment.hpp>

#include <pmacc/memory/buffers/Buffer.hpp>

namespace pmacc
{

template < typename T, std::size_t T_Dim >
class HostBuffer;

template < typename T, std::size_t T_Dim >
class DeviceBuffer;

namespace mem
{
namespace buffer
{

namespace device2host_detail
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
                               cudaMemcpyDeviceToHost,
                               cuda_stream));
}

template < typename T, typename T_DataAccessPolicy >
void copy(
    buffer::data::WriteGuard< HostBuffer<T, DIM1, T_DataAccessPolicy> > dst,
    buffer::data::ReadGuard< DeviceBuffer<T, DIM1, T_DataAccessPolicy> > src,
    DataSpace<DIM1> & size
)
{
    cudaStream_t cuda_stream = 0;
    CUDA_CHECK(cudaMemcpyAsync(dst.getBasePointer(),
                               src.getPointer(),
                               size[0] * sizeof (T),
                               cudaMemcpyDeviceToHost,
                               cuda_stream));
}

template < typename T, typename T_DataAccessPolicy >
void copy(
    buffer::data::WriteGuard< HostBuffer<T, DIM2, T_DataAccessPolicy> > dst,
    buffer::data::ReadGuard< DeviceBuffer<T, DIM2, T_DataAccessPolicy> > src,
    DataSpace<DIM2> & size
)
{
    cudaStream_t cuda_stream = 0;
    CUDA_CHECK(cudaMemcpy2DAsync(dst.getBasePointer(),
                                 dst.getDataSpace()[0] * sizeof (T), /*this is pitch*/
                                 src.getPointer(),
                                 src.getPitch(), /*this is pitch*/
                                 size[0] * sizeof (T),
                                 size[1],
                                 cudaMemcpyDeviceToHost,
                                 cuda_stream));
}

template < typename T, typename T_DataAccessPolicy >
void copy(
    buffer::data::WriteGuard< HostBuffer<T, DIM3, T_DataAccessPolicy> > dst,
    buffer::data::ReadGuard< DeviceBuffer<T, DIM3, T_DataAccessPolicy> > src,
    DataSpace<DIM3> & size
)
{
    cudaStream_t cuda_stream = 0;

    cudaPitchedPtr hostPtr;
    hostPtr.pitch = dst.getDataSpace()[0] * sizeof(T);
    hostPtr.ptr = dst.getBasePointer();
    hostPtr.xsize = dst.getDataSpace()[0] * sizeof(T);
    hostPtr.ysize = dst.getDataSpace()[1];

    cudaMemcpy3DParms params;
    params.srcArray = nullptr;
    params.srcPos = make_cudaPos(src.getOffset()[0] * sizeof(T),
                                 src.getOffset()[1],
                                 src.getOffset()[2]);
    params.srcPtr = src.getCudaPitched();

    params.dstArray = nullptr;
    params.dstPos = make_cudaPos(0, 0, 0);
    params.dstPtr = hostPtr;

    params.extent = make_cudaExtent(size[0] * sizeof(T),
                                    size[1],
                                    size[2]);
    params.kind = cudaMemcpyDeviceToHost;

    CUDA_CHECK(cudaMemcpy3DAsync(&params, cuda_stream))
}

} // namespace device2host_detail

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy
>
void
copy(
     WriteGuard< HostBuffer<T_Item, T_dim, T_DataAccessPolicy> > dst,
     ReadGuard< DeviceBuffer<T_Item, T_dim, T_DataAccessPolicy> > src
)
{
    Environment<>::task(
        []( auto dst, auto src, auto cuda_stream )
        {
            size_t current_size = src.size().get();

            dst.size().set(current_size);
            DataSpace<T_dim> devCurrentSize = src.size().data_space();

            if (src.data().is1D() && dst.data().is1D())
                device2host_detail::fast_copy(dst.data().getPointer(),
                                              src.data().getPointer(),
                                              devCurrentSize.productOfComponents());
            else
                device2host_detail::copy(dst.data(), src.data(), devCurrentSize);

            cuda_stream->sync();
        },
        TaskProperties::Builder()
            .label("pmacc::mem::buffer::copy(Host <= Device)"),
        std::move(dst),
        std::move(src),
        Environment<>::get().cuda_stream()
    );
}

} // namespace buffer

} // namespace mem

} // namespace pmacc


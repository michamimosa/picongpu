/* Copyright 2013-2018 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
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

#include "WaitForDevice.hpp"

namespace pmacc
{

template < typename T, std::size_t T_Dim >
class HostBuffer;

template < typename T, std::size_t T_Dim >
class DeviceBuffer;

namespace memory
{
namespace buffers
{

namespace device2host_detail
{

template < typename T >
void fast_copy(
    T * src,
    T * dst,
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

template < typename T >
void copy(
    HostBuffer<T, DIM1> & dst,
    DeviceBuffer<T, DIM1> & src,
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

template < typename T >
void copy(
    HostBuffer<T, DIM2> & dst,
    DeviceBuffer<T, DIM2> & src,
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

template < typename T >
void copy(
    HostBuffer<T, DIM3> & dst,
    DeviceBuffer<T, DIM3> & src,
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
    typename T,
    std::size_t T_Dim
>
void
copy(
    HostBuffer<T, T_Dim> & dst,
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
                device2host_detail::fast_copy(dst.getPointer(),
                                              src.getPointer(),
                                              devCurrentSize.productOfComponents());
            else
                device2host_detail::copy(dst, src, devCurrentSize);

            task_synchronize_stream(0);
        },
        TaskProperties::Builder()
            .label("copyDeviceToHost")
            .resources({
                dst.write(),
                dst.size_resource.write(),
                src.read(),
                src.size_resource.write(),
                cuda_resources::streams[0].write()
            })
    );
}

} // namespace buffers

} // namespace memory

} // namespace pmacc


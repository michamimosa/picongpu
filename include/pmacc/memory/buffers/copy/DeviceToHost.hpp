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

#include <pmacc/memory/buffers/HostBuffer.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>

namespace pmacc
{
    namespace mem
    {
        namespace buffer
        {
            namespace device2host_detail
            {
                template<typename T>
                void fast_copy(T* dst, T const* src, size_t size)
                {
                    CUDA_CHECK(cuplaMemcpyAsync(
                        dst,
                        src,
                        size * sizeof(T),
                        cuplaMemcpyDeviceToHost,
                        redGrapes::thread::current_cupla_stream));
                }

                template<typename T, typename T_DstDataAccessPolicy, typename T_SrcDataAccessPolicy>
                void copy(
                    host_buffer::data::WriteGuard<T, DIM1, T_DstDataAccessPolicy> const& dst,
                    device_buffer::data::ReadGuard<T, DIM1, T_SrcDataAccessPolicy> const& src,
                    DataSpace<DIM1> const& size)
                {
                    CUDA_CHECK(cuplaMemcpyAsync(
                        dst.getPointer(),
                        src.getPointer(),
                        size[0] * sizeof(T),
                        cuplaMemcpyDeviceToHost,
                        redGrapes::thread::current_cupla_stream));
                }

                template<typename T, typename T_DstDataAccessPolicy, typename T_SrcDataAccessPolicy>
                void copy(
                    host_buffer::data::WriteGuard<T, DIM2, T_DstDataAccessPolicy> const& dst,
                    device_buffer::data::ReadGuard<T, DIM2, T_SrcDataAccessPolicy> const& src,
                    DataSpace<DIM2> const& size)
                {
                    CUDA_CHECK(cuplaMemcpy2DAsync(
                        dst.getPointer(),
                        dst.getPitch(),
                        src.getPointer(),
                        src.getPitch(),
                        size[0] * sizeof(T),
                        size[1],
                        cuplaMemcpyDeviceToHost,
                        redGrapes::thread::current_cupla_stream));
                }

                template<typename T, typename T_DstDataAccessPolicy, typename T_SrcDataAccessPolicy>
                void copy(
                    host_buffer::data::WriteGuard<T, DIM3, T_DstDataAccessPolicy> const& dst,
                    device_buffer::data::ReadGuard<T, DIM3, T_SrcDataAccessPolicy> const& src,
                    DataSpace<DIM3> const& size)
                {
                    cuplaPitchedPtr hostPtr;
                    hostPtr.ptr = dst.getBasePointer();
                    hostPtr.pitch = src.getPitch();
                    hostPtr.xsize = src.getDataSpace()[0] * sizeof(T);
                    hostPtr.ysize = src.getDataSpace()[1];

                    cuplaMemcpy3DParms params;
                    params.srcArray = nullptr;
                    params.srcPos
                        = make_cuplaPos(src.getOffset()[0] * sizeof(T), src.getOffset()[1], src.getOffset()[2]);
                    params.srcPtr = src.getCudaPitched();

                    params.dstArray = nullptr;
                    params.srcPos
                        = make_cuplaPos(dst.getOffset()[0] * sizeof(T), dst.getOffset()[1], dst.getOffset()[2]);
                    params.dstPtr = hostPtr;

                    params.extent = make_cuplaExtent(size[0] * sizeof(T), size[1], size[2]);
                    params.kind = cuplaMemcpyDeviceToHost;

                    CUDA_CHECK(cuplaMemcpy3DAsync(&params, redGrapes::thread::current_cupla_stream))
                }

            } // namespace device2host_detail

            template<
                typename T_Item,
                std::size_t T_dim,
                typename T_DstDataAccessPolicy,
                typename T_SrcDataAccessPolicy>
            auto copy(
                host_buffer::WriteGuard<T_Item, T_dim, T_DstDataAccessPolicy> const& dst,
                device_buffer::ReadGuard<T_Item, T_dim, T_SrcDataAccessPolicy> const& src)
            {
                return Environment<>::task(
                    [](auto dst, auto src) {
                        dst.size().set(src.size().get());
                        DataSpace<T_dim> devCurrentSize = src.size().getCurrentDataSpace();

                        Environment<>::task(
                            [devCurrentSize](auto dst, auto src) {
                                if(src.is1D() && dst.is1D())
                                    device2host_detail::fast_copy(
                                        dst.getPointer(),
                                        src.getPointer(),
                                        devCurrentSize.productOfComponents());
                                else
                                {
                                    device2host_detail::copy(dst.write(), src.read(), devCurrentSize);
                                }
                            },
                            TaskProperties::Builder()
                                .label("cuplaMemcpyAsync(dst: Host, src: Device)")
                                .scheduling_tags({SCHED_CUPLA}),
                            dst.data(),
                            src.data());
                    },
                    TaskProperties::Builder().label("pmacc::mem::copy(dst: Host, src: Device)"),
                    dst.write(),
                    src.read());
            }

        } // namespace buffer

    } // namespace mem

} // namespace pmacc

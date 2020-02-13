/* Copyright 2016-2020 Alexander Grund, Michael Sippel
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

#include "HostDeviceBuffer.hpp"

namespace pmacc
{
namespace mem
{

template <typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy>
HostDeviceBuffer<T_Item, T_dim, T_DataAccessPolicy>::HostDeviceBuffer(DataSpace<T_dim> const & size, bool sizeOnDevice)
   : hostBuffer( (std::shared_ptr<HBuffer>)BufferResource<HostBufferType>( size ) )
   , deviceBuffer( (std::shared_ptr<DBuffer>)BufferResource<DeviceBufferType>( size, sizeOnDevice ) )
{}

template <typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy>
HostDeviceBuffer<T_Item, T_dim, T_DataAccessPolicy>::HostDeviceBuffer(
    BufferResource< DBuffer > otherDeviceBuffer,
    DataSpace<T_dim> const & size,
    bool sizeOnDevice)
    : hostBuffer( (std::shared_ptr<HBuffer>)BufferResource<HostBufferType>( size ) )
    , deviceBuffer( (std::shared_ptr<DBuffer>)BufferResource<DeviceBufferType>( otherDeviceBuffer, size, DataSpace<T_dim>(), sizeOnDevice ) )
{}

template<typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy>
HostDeviceBuffer<T_Item, T_dim, T_DataAccessPolicy>::HostDeviceBuffer(
    BufferResource<HBuffer> otherHostBuffer,
    DataSpace<T_dim> const & offsetHost,
    BufferResource<DBuffer> otherDeviceBuffer,
    DataSpace<T_dim> const & offsetDevice,
    GridLayout<T_dim> const size,
    bool sizeOnDevice)
  : hostBuffer( (std::shared_ptr<HBuffer>)BufferResource<HostBufferType>( otherHostBuffer, size, offsetHost ) )
   , deviceBuffer( (std::shared_ptr<DBuffer>)BufferResource<DeviceBufferType>( otherDeviceBuffer, size, offsetDevice, sizeOnDevice ) )
{}

template<typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy>
HostDeviceBuffer<T_Item, T_dim, T_DataAccessPolicy>::~HostDeviceBuffer()
{}

template < typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
BufferResource<HostBuffer<T_Item, T_dim, T_DataAccessPolicy>> HostDeviceBuffer<T_Item, T_dim, T_DataAccessPolicy>::getHostBuffer() const
{
    return hostBuffer;
}

template < typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
BufferResource<DeviceBuffer<T_Item, T_dim, T_DataAccessPolicy>> HostDeviceBuffer<T_Item, T_dim, T_DataAccessPolicy>::getDeviceBuffer() const
{
    return deviceBuffer;
}

template < typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
void HostDeviceBuffer<T_Item, T_dim, T_DataAccessPolicy>::reset(bool preserveData)
{
    deviceBuffer.data().reset(preserveData);
    hostBuffer.data().reset(preserveData);
}

template < typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
void HostDeviceBuffer<T_Item, T_dim, T_DataAccessPolicy>::hostToDevice()
{
    buffer::copy( deviceBuffer, hostBuffer );
}

template < typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
void HostDeviceBuffer<T_Item, T_dim, T_DataAccessPolicy>::deviceToHost()
{
    buffer::copy( hostBuffer, deviceBuffer );
}

} // namespac mem

} // namespace pmacc

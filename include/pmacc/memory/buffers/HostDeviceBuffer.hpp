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

#include <pmacc/types.hpp>
#include <pmacc/memory/buffers/HostBuffer.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>
#include <boost/type_traits.hpp>

#include <pmacc/memory/buffers/copy/DeviceToHost.hpp>

namespace pmacc
{
namespace mem
{
namespace host_device_buffer
{

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
struct HostDeviceBufferGuard
{
    using Item = T_Item;
    static constexpr std::size_t dim = T_dim;
    using DataAccessPolicy = T_DataAccessPolicy;
    using DataBoxType = DataBox< PitchedBox< Item, dim > >;

    
    //! todo: currently the size is stored doubled, in host buffer and device buffer
    //        and could possibly be shared (using only one DeviceBufferSize for both buffers)
    host_buffer::WriteGuard< T_Item, T_dim, T_DataAccessPolicy > host_buf;
    device_buffer::WriteGuard< T_Item, T_dim, T_DataAccessPolicy > device_buf;
};

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
struct HostDeviceBufferResource
{
    using Item = T_Item;
    static constexpr std::size_t dim = T_dim;
    using DataAccessPolicy = T_DataAccessPolicy;
    using DataBoxType = DataBox< PitchedBox< Item, dim > >;

    HostDeviceBufferResource(
        DataSpace< dim > capacity,
        bool use_vector_as_base = false
    ) :
        host( capacity ),
        device( use_vector_as_base )
    {}

    auto make_guard( bool size_on_device = false )
    {
        return HostDeviceBufferGuard< Item, dim, DataAccessPolicy > {
            host.make_guard(), device.make_guard( size_on_device )
        };
    }

protected:
    host_buffer::HostBufferResource<
        Item,
        dim,
        DataAccessPolicy
    > host;

    device_buffer::DeviceBufferResource<
        Item,
        dim,
        DataAccessPolicy
    > device;
};

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
struct ReadGuard
    : protected HostDeviceBufferGuard< T_Item, T_dim, T_DataAccessPolicy >
{
    auto read() const noexcept { return *this; }

    auto host() const noexcept { return this->host_buf.read(); }
    auto device() const noexcept { return this->device_buf.read(); }

    auto sub_area(
        DataSpace< T_dim > offset,
        DataSpace< T_dim > data_space
    ) const
    {
        return ReadGuard(HostDeviceBufferGuard< T_Item, T_dim, T_DataAccessPolicy >{
                this->host_buf.sub_area( offset, data_space ),
                this->device_buf.sub_area( offset, data_space )
        });
    }

protected:
    ReadGuard( HostDeviceBufferGuard< T_Item, T_dim, T_DataAccessPolicy > const & b )
        : HostDeviceBufferGuard<T_Item, T_dim, T_DataAccessPolicy>( b ) {}
};

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
struct WriteGuard
    : protected ReadGuard< T_Item, T_dim, T_DataAccessPolicy >
{
    operator ReadGuard< T_Item, T_dim, T_DataAccessPolicy >() const noexcept { return this->read(); }
    auto write() const noexcept { return *this; }

    auto host() const noexcept { return this->host_buf.write(); }
    auto device() const noexcept { return this->device_buf.write(); }

    auto sub_area(
        DataSpace< T_dim > offset,
        DataSpace< T_dim > data_space
    ) const
    {
        return WriteGuard(HostDeviceBufferGuard< T_Item, T_dim, T_DataAccessPolicy >{
            this->host_buf.sub_area( offset, data_space ),
            this->device_buf.sub_area( offset, data_space )
        });
    }

    void deviceToHost()
    {
        pmacc::mem::buffer::copy( host(), device() );
    }

protected:
    WriteGuard( HostDeviceBufferGuard< T_Item, T_dim, T_DataAccessPolicy> const & b )
        : ReadGuard< T_Item, T_dim, T_DataAccessPolicy >( b ) {}
};

} // namespace host_device_buffer

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
struct HostDeviceBuffer
    : host_device_buffer::WriteGuard< T_Item, T_dim, T_DataAccessPolicy >
{    
    HostDeviceBuffer(
        DataSpace< T_dim > capacity,
        bool size_on_device = false
    ) :
        host_device_buffer::WriteGuard<
            T_Item,
            T_dim,
            T_DataAccessPolicy
        >(
            host_device_buffer::HostDeviceBufferGuard<
                T_Item,
                T_dim,
                T_DataAccessPolicy
            >{
                HostBuffer< T_Item, T_dim, T_DataAccessPolicy >( capacity ),
                DeviceBuffer< T_Item, T_dim, T_DataAccessPolicy >( capacity, size_on_device )
            }
        )
    {}

    HostDeviceBuffer(
        device_buffer::WriteGuard< T_Item, T_dim, T_DataAccessPolicy > other_device_buffer,
        DataSpace< T_dim > capacity
    ) :
        host_device_buffer::WriteGuard<
            T_Item,
            T_dim,
            T_DataAccessPolicy
        >(
            host_device_buffer::HostDeviceBufferGuard<
                T_Item,
                T_dim,
                T_DataAccessPolicy
            >{
                HostBuffer< T_Item, T_dim, T_DataAccessPolicy >( capacity ),
                other_device_buffer
            }
        )
    {}
};

} // namespace mem

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
using HostDeviceBuffer = mem::HostDeviceBuffer< T_Item, T_dim, T_DataAccessPolicy >;

} // namespace pmacc

namespace redGrapes
{
namespace trait
{

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy
>
struct BuildProperties<
    pmacc::mem::host_device_buffer::ReadGuard< T_Item, T_dim, T_DataAccessPolicy >
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::host_device_buffer::ReadGuard< T_Item, T_dim, T_DataAccessPolicy > const & buf
    )
    {
        builder.add( buf.host() );
        builder.add( buf.device() );
    }
};

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy
>
struct BuildProperties<
    pmacc::mem::host_device_buffer::WriteGuard< T_Item, T_dim, T_DataAccessPolicy >
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::host_device_buffer::ReadGuard< T_Item, T_dim, T_DataAccessPolicy > const & buf
    )
    {
        builder.add( buf.host() );
        builder.add( buf.device() );
    }
};

} // namespace trait

} // namespace redGrapes


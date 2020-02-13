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
#include <pmacc/memory/buffers/HostBufferIntern.hpp>
#include <pmacc/memory/buffers/DeviceBufferIntern.hpp>
#include <pmacc/memory/buffers/CopyDeviceToHost.hpp>
#include <pmacc/memory/buffers/CopyHostToDevice.hpp>
#include <boost/type_traits.hpp>

namespace pmacc
{
namespace mem
{

    /** Buffer that contains a host and device buffer and allows synchronizing those 2 */
    template<typename T_Type, std::size_t T_dim, typename T_DataAccessPolicy = rg::access::IOAccess >
    class HostDeviceBuffer
    {
        typedef HostBufferIntern<T_Type, T_dim, T_DataAccessPolicy> HostBufferType;
        typedef DeviceBufferIntern<T_Type, T_dim, T_DataAccessPolicy> DeviceBufferType;

    public:
        using ValueType = T_Type;
        typedef HostBuffer<T_Type, T_dim, T_DataAccessPolicy> HBuffer;
        typedef DeviceBuffer<T_Type, T_dim, T_DataAccessPolicy> DBuffer;

        typedef typename HostBufferType::DataBoxType DataBoxType;
        PMACC_CASSERT_MSG(DataBoxTypes_must_match, boost::is_same<DataBoxType, typename DeviceBufferType::DataBoxType>::value);

        /**
         * Constructor that creates the buffers with the given size
         *
         * @param size DataSpace representing buffer size
         * @param sizeOnDevice if true, internal buffers must store their
         *        size additionally on the device
         *        (as we keep this information coherent with the host, it influences
         *        performance on host-device copies, but some algorithms on the device
         *        might need to know the size of the buffer)
         */
        HostDeviceBuffer(DataSpace<T_dim> const & size, bool sizeOnDevice = false);

        /**
         * Constructor that reuses the given device buffer instead of creating an own one.
         * Sizes should match. If size is smaller than the buffer size, then only the part near the origin is used.
         * Passing a size bigger than the buffer is undefined.
         */
        HostDeviceBuffer(BufferResource<DBuffer> otherDeviceBuffer,
                         DataSpace<T_dim> const & size,
                         bool sizeOnDevice = false);


        
        /**
         * Constructor that reuses the given buffers instead of creating own ones.
         * The data from [offset, offset+size) is used
         * Passing a size bigger than the buffer (minus the offset) is undefined.
         */
        HostDeviceBuffer(
                   BufferResource<HBuffer> otherHostBuffer,
                   DataSpace<T_dim> const & offsetHost,
                   BufferResource<DBuffer> otherDeviceBuffer,
                   DataSpace<T_dim> const & offsetDevice,
                   GridLayout<T_dim> const size,
                   bool sizeOnDevice = false);

        HINLINE virtual ~HostDeviceBuffer();

        /**
         * Returns the internal data buffer on host side
         *
         * @return internal HBuffer
         */
        HINLINE BufferResource<HBuffer> getHostBuffer() const;

        /**
         * Returns the internal data buffer on device side
         *
         * @return internal DBuffer
         */
        HINLINE BufferResource<DBuffer> getDeviceBuffer() const;

        /**
         * Resets both internal buffers.
         *
         * See DeviceBuffer::reset and HostBuffer::reset for details.
         *
         * @param preserveData determines if data on internal buffers should not be erased
         */
        void reset(bool preserveData = true);

        /**
         * Asynchronously copies data from internal host to internal device buffer.
         *
         */
        HINLINE void hostToDevice();

        /**
         * Asynchronously copies data from internal device to internal host buffer.
         */
        HINLINE void deviceToHost();

    private:
        BufferResource< HBuffer > hostBuffer;
        BufferResource< DBuffer > deviceBuffer;
    };

    namespace host_device_buffer
    {
        template < typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
        struct ReadGuard : protected std::shared_ptr< HostDeviceBuffer<T_Item, T_dim, T_DataAccessPolicy> >
        {
            auto read() { return *this; }
            HINLINE auto getHostBuffer() const { return this->get()->getHostBuffer().read(); }
            HINLINE auto getDeviceBuffer() const { return this->get()->getDeviceBuffer().read(); }

        protected:
            ReadGuard( std::shared_ptr< HostDeviceBuffer<T_Item, T_dim, T_DataAccessPolicy> > obj )
                : std::shared_ptr< HostDeviceBuffer< T_Item, T_dim, T_DataAccessPolicy > >( obj )
            {}
        };

        template < typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
        struct WriteGuard : ReadGuard< T_Item, T_dim, T_DataAccessPolicy >
        {
            auto write() { return *this; }
            HINLINE auto getHostBuffer() const { return this->get()->getHostBuffer().write(); }
            HINLINE auto getDeviceBuffer() const { return this->get()->getDeviceBuffer().write(); }

            HINLINE void hostToDevice() { buffer::copy(getDeviceBuffer(), getHostBuffer()); }
            HINLINE void deviceToHost() { buffer::copy(getHostBuffer(), getDeviceBuffer()); }

            void reset(bool preserveData = true) { this->get()->reset(preserveData); }

        protected:
            WriteGuard( std::shared_ptr< HostDeviceBuffer<T_Item, T_dim, T_DataAccessPolicy> > obj )
                : ReadGuard< T_Item, T_dim, T_DataAccessPolicy >( obj )
            {}
        };
    } // namespace host_device_buffer


    template< typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
    struct BufferResource< HostDeviceBuffer< T_Item, T_dim, T_DataAccessPolicy > > : host_device_buffer::WriteGuard< T_Item, T_dim, T_DataAccessPolicy >
    {
        template < typename... Args >
        BufferResource( Args&&... args )
            : host_device_buffer::WriteGuard< T_Item, T_dim, T_DataAccessPolicy >( std::make_shared< HostDeviceBuffer< T_Item, T_dim, T_DataAccessPolicy > >( std::forward<Args>(args)... ) )
        {}

        BufferResource( std::shared_ptr< HostDeviceBuffer< T_Item, T_dim, T_DataAccessPolicy > > obj )
            : host_device_buffer::WriteGuard< T_Item, T_dim, T_DataAccessPolicy >( obj )
        {}
    };

} // namespace mem

} // namespace pmacc

#include "pmacc/memory/buffers/HostDeviceBuffer.tpp"

namespace redGrapes
{
namespace trait
{

template < typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
struct BuildProperties< pmacc::mem::host_device_buffer::ReadGuard<T_Item, T_dim, T_DataAccessPolicy> >
{
    template < typename Builder >
    static void build(Builder & builder, pmacc::mem::host_device_buffer::ReadGuard<T_Item, T_dim, T_DataAccessPolicy> const & buf)
    {
        BuildProperties< decltype(buf.getHostBuffer()) >::build( builder, buf.getHostBuffer() );
        BuildProperties< decltype(buf.getDeviceBufer()) >::build( builder, buf.getDeviceBuffer() );
    }
};

template < typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
struct BuildProperties< pmacc::mem::host_device_buffer::WriteGuard<T_Item, T_dim, T_DataAccessPolicy> >
{
    template < typename Builder >
    static void build(Builder & builder, pmacc::mem::host_device_buffer::WriteGuard<T_Item, T_dim, T_DataAccessPolicy> const & buf)
    {
        BuildProperties< decltype(buf.getHostBuffer()) >::build( builder, buf.getHostBuffer() );
        BuildProperties< decltype(buf.getDeviceBufer()) >::build( builder, buf.getDeviceBuffer() );
    }
};

} // namespace trait

} // namespace redGrapes


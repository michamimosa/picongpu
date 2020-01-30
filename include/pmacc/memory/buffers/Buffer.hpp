/* Copyright 2013-2020 Heiko Burau, Rene Widera, Benjamin Worpitz, Michael Sippel
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"
#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/types.hpp"

#include <redGrapes/resource/fieldresource.hpp>
#include <redGrapes/resource/ioresource.hpp>

#include <limits>

namespace rg = redGrapes;

#include <share/thirdParty/akrzemi/optional/optional.hpp>
namespace std
{
    using optional = experimental::optional;
} // namespace std

namespace pmacc
{
namespace mem
{
namespace buffer
{
   
template< std::size_t T_Dim >
struct Access
{
    std::optional< rg::access::IO > size;
    std::optional< rg::access::Field< T_Dim > > data;

    static bool is_serial( Access const & a, Access const & b )
    {
        if( a.data && b.data )
            if( rg::access::Field<T_Dim>::is_serial( *a.data, *b.data ) )
                return true;

        if( a.size && b.size )
            if( rg::access::IO::is_serial( *a.size, *b.size ) )
                return true;

        return false;
    }

    bool is_superset_of( Access const & sub )
    {
        if( !size )
        {
            if( sub.size )
                return false;
        }
        else if( sub.size && !size->is_superset_of(sub.size) )
            return false;

        if( !data )
        {
            if( sub.data )
                return false;
        }
        else if( sub.data && !data->is_superset_of(sub.data) )
            return false;

        return true;
    }

    static auto access( rg::access::Field< T_Dim > data ) { return Access { std::nullopt, data }; }
    static auto access_size_read() { return Access { rg::access::IO::read, std::nullopt }; }
    static auto access_size_write() { return Access { rg::access::IO::write, std::nullopt }; }
    static auto access_write() { return access_size_write( rg::access::Field(rg::access::IO::write) ); }
};

namespace size
{
    template < typename Buffer >
    struct ReadGuard : rg::SharedResourceObject< Buffer, Access >
    {
        operator ResourceAccess() const noexcept
        {
            return this->make_access(BufferAccess{ rg::access::IO::read, std::nullopt });
        }

        ReadGuard read() const noexcept { return *this; }
        std::size_t get() const { return this->obj->get_size(); }
    };

    template < typename Buffer >
    struct WriteGuard : ReadGuard< Buffer >
    {
        operator ResourceAccess() const noexcept
        {
            return this->make_access(BufferAccess{ rg::access::IO::write, std::nullopt });
        }

        WriteGuard write() { return *this; }
        void set( size_t const new_size ) const { this->obj->set_size( new_size ); }
    };
} // namespace size

namespace data
{
    template < typename Buffer >
    struct ReadGuard : rg::SharedResourceObject< Buffer, Access >
    {
        operator ResourceAccess() const noexcept
        {
            return this->make_access(BufferAccess{ std::nullopt, rg::access::Field<Buffer::T_Dim>(rg::access::IO::read) });
        }

        ReadGuard read() const noexcept { return *this; }

        using Item = typename Buffer::Item;
        using DataBox = typename Buffer::DataBox;

        bool is1D() const { return buffer->obj->is1D(); }
        Item const * getPointer() const { return this->obj->getPointer(); }
        Item const * getBasePointer() const { return this->obj->getBasePointer(); }
        DataBox const getDataBox() const { return->obj->getDataBox(); }
    };

    template < typename Buffer >
    struct WriteGuard : ReadGuard< Buffer >
    {
        operator ResourceAccess() const noexcept
        {
            return this->make_access(BufferAccess{ std::nullopt, rg::access::Field<Buffer::T_Dim>(rg::access::IO::write) });
        }

        WriteGuard write() const noexcept { return *this; }

        using Item = typename Buffer::Item;
        using DataBox = typename Buffer::DataBox;

        void reset() const { buffer->obj->reset(); }
        void fill(Buffer::Item const & item) const { this->obj->fill(item); }
        Item * getPointer() const { return this->obj->getPointer();  }
        Item * getBasePointer() const { return this->obj->getBasePointer(); }
        DataBox getDataBox() const { return->obj->getDataBox(); }
    };
} // namespace data

template < typename Buffer >
struct ReadGuard : rg::SharedResourceObject< Buffer, BufferAccess >
{
    ReadGuard read() const noexcept { return *this; }

    operator ResourceAccess() const noexcept
    {
        return this->make_access(BufferAccess{ rg::access::IOAccess::write, rg::access::FieldAccess<Buffer::T_Dim>(rg::access::IOAccess::write) });
    }

    auto data() const noexcept { return data::ReadGuard( *this ); }
    auto size() const noexcept { return size::ReadGuard( *this ); }
};

template < typename Buffer >
struct WriteGuard : ReadGuard< Buffer >
{
    WriteGuard write() const noexcept { return *this; }
    operator ResourceAccess() const noexcept
    {
        return this->make_access(BufferAccess{ redGrapes::access::IOAccess::write, this->m_area });
    }

    auto data() const noexcept { return data::WriteGuard( *this ); }
    auto size() const noexcept { return size::WriteGuard( *this ); }
};

} // namespace buffer

template < typename Buffer >
struct BufferResource : buffer::WriteGuard
{
    template < typename... Args >
    BufferResource( Args&&... arg )
        : buffer::WriteGuard(
              rg::SharedResourceObject< Buffer, buffer::Access >( std::make_shared<Buffer>(std::forward<Args>(args)...) ))
    {
        this->resource = this->obj->resource;
    }
};

} // namespace mem

    /**
     * Minimal function description of a buffer,
     *
     * @tparam TYPE data type stored in the buffer
     * @tparam DIM dimension of the buffer (1-3)
     */
    template <
        typename T_Item,
        std::size_t T_dim >
    class Buffer
        : std::enable_shared_from_this< Buffer<Item, T_dim> >
        , rg::Resource< buffer::Access< T_dim > >
    {
    public:
        using Item = T_Item;
        static constexpr size_t dim = T_dim;

        typedef DataBox< PitchedBox<Item, dim> > DataBoxType;

        /** constructor
         *
         * @param size extent for each dimension (in elements)
         *             if the buffer is a view to an existing buffer the size
         *             can be less than `physicalMemorySize`
         * @param physicalMemorySize size of the physical memory (in elements)
         */
        Buffer(DataSpace< T_dim > size,
               DataSpace< T_dim > physicalMemorySize,
               rg::Resource< BufferAccess<Item, T_dim> > resource)
            : data_space( size )
            , data1D( true )
            , current_size( nullptr )
            , m_physicalMemorySize( physicalMemorySize )
            , resource_id( resource )
        {
            Environment<>::task(
                [obj=shared_from_this()] {
                    CUDA_CHECK(cudaMallocHost((void**)&current_size, sizeof(size_t)));
                    *current_size = size.productOfComponents();
                },
                TaskProperties::Builder()
                    .label("Buffer::Buffer()")
                    .resources({ access_size_write() })
            );
        }

        /**
         * destructor
         */
        ~Buffer()
        {
            Environment<>::task(
                [obj=shared_from_this()] {
                    CUDA_CHECK_NO_EXCEPT(cudaFreeHost(current_size));
                },
                TaskProperties::Builder()
                    .label("Buffer::~Buffer()")
                    .resources({ access_size_write() })
            );
        }

        /*! Get base pointer to memory
         * @return pointer to this buffer in memory
         */
        virtual Item const * getBasePointer() const = 0;
        virtual Item * getBasePointer() = 0;

        /*! Get pointer that includes all offsets
         * @return pointer to a point in a memory array
         */
        virtual Item const * getPointer() const = 0;
        virtual Item * getPointer() = 0;

        /*! Get max spread (elements) of any dimension
         * @return spread (elements) per dimension
         */
        virtual DataSpace<DIM> getDataSpace() const
        {
            return data_space;
        }

        /** get size of the physical memory (in elements)
         */
        DataSpace<DIM> getPhysicalMemorySize() const
        {
            return m_physicalMemorySize;
        }

        virtual DataSpace<DIM> getCurrentDataSpace()
        {
            return getCurrentDataSpace(getCurrentSize());
        }

        /*! Spread of memory per dimension which is currently used
         * @return if DIM == DIM1 than return count of elements (x-direction)
         * if DIM == DIM2 than return how many lines (y-direction) of memory is used
         * if DIM == DIM3 than return how many slides (z-direction) of memory is used
         */
        virtual DataSpace<DIM> getCurrentDataSpace(size_t currentSize)
        {
            DataSpace<DIM> tmp;
            int64_t current_size = static_cast<int64_t>(currentSize);

            //!\todo: current size can be changed if it is a DeviceBuffer and current size is on device
            //call first get current size (but const not allow this)

            if (DIM == DIM1)
            {
                tmp[0] = current_size;
            }
            if (DIM == DIM2)
            {
                if (current_size <= data_space[0])
                {
                    tmp[0] = current_size;
                    tmp[1] = 1;
                } else
                {
                    tmp[0] = data_space[0];
                    tmp[1] = (current_size+data_space[0]-1) / data_space[0];
                }
            }
            if (DIM == DIM3)
            {
                if (current_size <= data_space[0])
                {
                    tmp[0] = current_size;
                    tmp[1] = 1;
                    tmp[2] = 1;
                } else if (current_size <= (data_space[0] * data_space[1]))
                {
                    tmp[0] = data_space[0];
                    tmp[1] = (current_size+data_space[0]-1) / data_space[0];
                    tmp[2] = 1;
                } else
                {
                    tmp[0] = data_space[0];
                    tmp[1] = data_space[1];
                    tmp[2] = (current_size+(data_space[0] * data_space[1])-1) / (data_space[0] * data_space[1]);
                }
            }

            return tmp;
        }

        /*! returns the current size (count of elements)
         * @return current size
         */
        virtual auto get_size()
        {
            return pmacc::Environment<>::task(
                [obj=shared_from_this()] {
                    return *(obj->current_size);
                },
                TaskProperties::Builder()
                    .label("Buffer::get_size()")
                    .resources({ access_size_read() })
            );
        }

        /*! sets the current size (count of elements)
         * @param newsize new current size
         */
        virtual void set_size( size_t const newsize )
        {
            pmacc::Environment<>::task(
                [obj=shared_from_this()] {
                    PMACC_ASSERT(static_cast<size_t>(newsize) <= static_cast<size_t>(data_space.productOfComponents()));
                    *(buffer->obj->current_size) = newsize;
                },
                TaskProperties::Builder()
                    .label("Buffer::set_size(" + std::to_string(newsize) + ")")
                    .resources({ access_size_write() })
);
        }

        virtual void reset( bool preserveData = false ) = 0;
        virtual void fill( Item const & value ) = 0;

        virtual DataBox< PitchedBox<Item, T_dim> > getDataBox() = 0;

        inline bool is1D() const noexcept
        {
            return data1D;
        }

    protected:
        /*! Check if my DataSpace is greater than other.
         * @param other other DataSpace
         * @return true if my DataSpace (one dimension) is greater than other, false otherwise
         */
        virtual bool isMyDataSpaceGreaterThan(DataSpace<DIM> other)
        {
            return !other.isOneDimensionGreaterThan(data_space);
        }

        DataSpace<DIM> data_space;
        DataSpace<DIM> m_physicalMemorySize;

        size_t *current_size;

        bool data1D;
    };

} // namespace pmacc

namespace redGrapes
{
namespace trait
{

template< typename Buffer >
TRAIT_BUILD_RESOURCE_PROPERTIES( buffer::size::ReadGuard<Buffer> );
template< typename Buffer >
TRAIT_BUILD_RESOURCE_PROPERTIES( buffer::size::WriteGuard<Buffer> );
template< typename Buffer >
TRAIT_BUILD_RESOURCE_PROPERTIES( buffer::data::ReadGuard<Buffer> );
template< typename Buffer >
TRAIT_BUILD_RESOURCE_PROPERTIES( buffer::data::WriteGuard<Buffer> );
template< typename Buffer >
TRAIT_BUILD_RESOURCE_PROPERTIES( buffer::ReadGuard<Buffer> );
template< typename Buffer >
TRAIT_BUILD_RESOURCE_PROPERTIES( buffer::WriteGuard<Buffer> );

} // namespace trait
}


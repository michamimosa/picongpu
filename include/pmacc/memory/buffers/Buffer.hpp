/* Copyright 2013-2020 Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Michael Sippel
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

#include <akrzemi/optional.hpp>

namespace std
{

template < typename T >
using optional = experimental::optional< T >;
auto nullopt = experimental::nullopt;

} // namespace std

namespace pmacc
{
namespace mem
{
namespace buffer
{
   
template< std::size_t T_dim >
struct Access
{
    std::optional< rg::access::IOAccess > size;
    std::optional< rg::access::FieldAccess< T_dim > > data;

    friend bool operator==(Access const & a, Access const & b )
    {
        return a.size == b.size && a.data == b.data;
    }

    friend std::ostream& operator<<( std::ostream& out, Access const & a )
    {
        out << "BufferAccess{" << std::endl;
        if( a.size )
            out << "size: " << *a.size << ";" << std::endl;
        if( a.data )
            out << "data: " << *a.data << ";" << std::endl;
        out << "}";

        return out;
    }
    
    static bool is_serial( Access const & a, Access const & b )
    {
        if( a.data && b.data )
            if( rg::access::FieldAccess<T_dim>::is_serial( *a.data, *b.data ) )
                return true;
        if( a.size && b.size )
            if( rg::access::IOAccess::is_serial( *a.size, *b.size ) )
                return true;
        return false;
    }

    bool is_superset_of( Access const & sub ) const
    {
        if( !data && sub.data )
            return false;
        else if( sub.data )
            return data->is_superset_of(*sub.data);

        if( !size && sub.size )
            return false;
        else if( sub.size && !size->is_superset_of(*sub.size) )
            return false;

        return true;
    }

    static auto data_read() { return Access{ std::nullopt, rg::access::FieldAccess<T_dim>(rg::access::IOAccess::read) }; }
    static auto data_write() { return Access{ std::nullopt, rg::access::FieldAccess<T_dim>(rg::access::IOAccess::write) }; }
    static auto size_read() { return Access { rg::access::IOAccess(rg::access::IOAccess::read), std::nullopt }; }
    static auto size_write() { return Access { rg::access::IOAccess(rg::access::IOAccess::write), std::nullopt }; }
    static auto write() { return Access{
            rg::access::IOAccess(rg::access::IOAccess::write),
            rg::access::FieldAccess< T_dim >(rg::access::IOAccess(rg::access::IOAccess::write)) }; }
    static auto read() { return Access{
            rg::access::IOAccess(rg::access::IOAccess::read),
            rg::access::FieldAccess< T_dim >(rg::access::IOAccess(rg::access::IOAccess::read)) }; }
};

template < typename Buffer >
struct ReadGuard;
template < typename Buffer >
struct WriteGuard;

namespace size
{
    template < typename Buffer >
    struct ReadGuard : rg::SharedResourceObject< Buffer, Access< Buffer::dim > >
    {
        operator rg::ResourceAccess() const noexcept
        {
            return this->make_access(Access<Buffer::dim>::size_read());
        }

        ReadGuard read() const noexcept { return *this; }
        std::size_t get() const { return this->obj->get_size(); }

    protected:
        friend class ::pmacc::mem::buffer::ReadGuard< Buffer >;
        ReadGuard( rg::SharedResourceObject< Buffer, Access< Buffer::dim > > obj )
            : rg::SharedResourceObject< Buffer, Access< Buffer::dim > >( obj )
        {}        
    };

    template < typename Buffer >
    struct WriteGuard : ReadGuard< Buffer >
    {
        operator rg::ResourceAccess() const noexcept
        {
            return this->make_access(Access<Buffer::dim>::size_write());
        }

        WriteGuard write() { return *this; }
        void set( size_t const new_size ) const { this->obj->set_size( new_size ); }

    protected:
        friend class ::pmacc::mem::buffer::WriteGuard< Buffer >;
        WriteGuard( rg::SharedResourceObject< Buffer, Access< Buffer::dim > > obj )
            : ReadGuard<Buffer>( obj )
        {}
    };
} // namespace size

namespace data
{
    template < typename Buffer >
    struct ReadGuard : rg::SharedResourceObject< Buffer, Access< Buffer::dim > >
    {
        operator rg::ResourceAccess() const noexcept
        {
            return this->make_access(Access<Buffer::dim>::data_read());
        }

        ReadGuard read() const noexcept { return *this; }

        using Item = typename Buffer::Item;
        using DataBox = typename Buffer::DataBoxType;

        bool is1D() const { return this->obj->is1D(); }
        Item const * getPointer() const { return this->obj->getPointer(); }
        Item const * getBasePointer() const { return this->obj->getBasePointer(); }
        DataBox const getDataBox() const { return this->obj->getDataBox(); }

    protected:
        friend class ::pmacc::mem::buffer::ReadGuard< Buffer >;
        ReadGuard( rg::SharedResourceObject< Buffer, Access< Buffer::dim > > obj )
            : rg::SharedResourceObject< Buffer, Access< Buffer::dim > >( obj )
        {}
    };

    template < typename Buffer >
    struct WriteGuard : ReadGuard< Buffer >
    {
        operator rg::ResourceAccess() const noexcept
        {
            return this->make_access(Access<Buffer::dim>::data_write());
        }

        WriteGuard write() const noexcept { return *this; }

        using Item = typename Buffer::Item;
        using DataBox = typename Buffer::DataBoxType;

        void reset() const { this->obj->reset(); }
        void fill(Item const & item) const { this->obj->fill(item); }
        Item * getPointer() const { return this->obj->getPointer();  }
        Item * getBasePointer() const { return this->obj->getBasePointer(); }
        DataBox getDataBox() const { return this->obj->getDataBox(); }

    protected:
        friend class ::pmacc::mem::buffer::WriteGuard< Buffer >;
        WriteGuard( rg::SharedResourceObject< Buffer, Access< Buffer::dim > > obj )
            : ReadGuard<Buffer>( obj )
        {}
    };
} // namespace data

template < typename Buffer >
struct ReadGuard : public rg::SharedResourceObject< Buffer, Access< Buffer::dim > >
{
    ReadGuard read() const noexcept { return *this; }

    operator rg::ResourceAccess() const noexcept
    {
        return this->make_access(Access<Buffer::dim>::read());
    }

    auto data() const noexcept { return data::ReadGuard<Buffer>( (rg::SharedResourceObject<Buffer, Access<Buffer::dim>>)*this ); }
    auto size() const noexcept { return size::ReadGuard<Buffer>( (rg::SharedResourceObject<Buffer, Access<Buffer::dim>>)*this ); }

protected:
    ReadGuard( std::shared_ptr< Buffer > obj )
        : rg::SharedResourceObject<Buffer, Access< Buffer::dim >>( obj )
    {}
};

template < typename Buffer >
struct WriteGuard : ReadGuard< Buffer >
{
    WriteGuard write() const noexcept { return *this; }
    operator rg::ResourceAccess() const noexcept
    {
        return this->make_access(Access<Buffer::dim>::write());
    }

    auto data() const noexcept { return data::WriteGuard<Buffer>( *this ); }
    auto size() const noexcept { return size::WriteGuard<Buffer>( *this ); }

protected:
    WriteGuard( std::shared_ptr<Buffer> obj )
        : ReadGuard<Buffer>( obj )
    {}
};

} // namespace buffer

template < typename Buffer >
struct BufferResource : buffer::WriteGuard< Buffer >
{
    template < typename... Args >
    BufferResource( Args&&... args )
        : buffer::WriteGuard<Buffer>( std::make_shared<Buffer>( std::forward<Args>(args)... ) )
    {
        *((rg::ResourceBase*)this) = (rg::ResourceBase) *this->obj;
    }
};

    /**
     * Minimal function description of a buffer,
     *
     * @tparam TYPE data type stored in the buffer
     * @tparam DIM dimension of the buffer (1-3)
     */
    template <
        typename T_Item,
        std::size_t T_dim
    >
    class Buffer
        : public std::enable_shared_from_this< Buffer<T_Item, T_dim> >
        , public rg::Resource< buffer::Access< T_dim > >
    {
    protected:
        template < typename Derived >
        std::shared_ptr< Derived > shared_from_base()
        {
            return std::static_pointer_cast< Derived >( this->shared_from_this() );
        }

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
        Buffer(DataSpace< dim > size,
               DataSpace< dim > physicalMemorySize,
               rg::Resource< buffer::Access< dim > > resource)
            : rg::Resource< buffer::Access< dim > >( resource )
            , data_space( size )
            , data1D( true )
            , current_size( nullptr )
            , m_physicalMemorySize( physicalMemorySize )
        {
            Environment<>::task(
                [this, size] {
                    CUDA_CHECK(cudaMallocHost((void**)&this->current_size, sizeof(size_t)));
                    *this->current_size = size.productOfComponents();
                },
                TaskProperties::Builder()
                    .label("Buffer::Buffer()")
                    .resources({ this->make_access(buffer::Access<dim>::size_write()) }));
        }

        /**
         * destructor
         */
        virtual ~Buffer()
        {
            CUDA_CHECK_NO_EXCEPT(cudaFreeHost(this->current_size));
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
        virtual DataSpace<dim> getDataSpace() const
        {
            return data_space;
        }

        /** get size of the physical memory (in elements)
         */
        DataSpace<dim> getPhysicalMemorySize() const
        {
            return m_physicalMemorySize;
        }

        virtual DataSpace<dim> getCurrentDataSpace()
        {
            return getCurrentDataSpace(get_size());
        }

        /*! Spread of memory per dimension which is currently used
         * @return if DIM == DIM1 than return count of elements (x-direction)
         * if DIM == DIM2 than return how many lines (y-direction) of memory is used
         * if DIM == DIM3 than return how many slides (z-direction) of memory is used
         */
        virtual DataSpace<dim> getCurrentDataSpace(size_t currentSize)
        {
            DataSpace<dim> tmp;
            int64_t current_size = static_cast<int64_t>(currentSize);

            //!\todo: current size can be changed if it is a DeviceBuffer and current size is on device
            //call first get current size (but const not allow this)

            if (dim == DIM1)
            {
                tmp[0] = current_size;
            }
            if (dim == DIM2)
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
            if (dim == DIM3)
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
        virtual size_t get_size()
        {
            return Environment<>::task(
                [obj=this->shared_from_this()] {
                    return *(obj->current_size);
                },
                TaskProperties::Builder()
                    .label("Buffer::get_size()")
                    .resources({ this->make_access(buffer::Access<dim>::size_read()) })
            ).get();
        }

        /*! sets the current size (count of elements)
         * @param newsize new current size
         */
        virtual void set_size( size_t const new_size )
        {
            Environment<>::task(
                [obj=this->shared_from_this(), new_size] {
                    PMACC_ASSERT(static_cast<size_t>(new_size) <= static_cast<size_t>(obj->data_space.productOfComponents()));
                    *(obj->current_size) = new_size;
                },
                TaskProperties::Builder()
                    .label("Buffer::set_size(" + std::to_string(new_size) + ")")
                    .resources({ this->make_access(buffer::Access<dim>::size_write()) })
            );
        }

        virtual void reset( bool preserveData = false ) = 0;
        virtual void fill( Item const & value ) = 0;

        virtual DataBox< PitchedBox<Item, dim> > getDataBox() = 0;

        inline bool is1D() const noexcept
        {
            return data1D;
        }

    protected:
        /*! Check if my DataSpace is greater than other.
         * @param other other DataSpace
         * @return true if my DataSpace (one dimension) is greater than other, false otherwise
         */
        virtual bool isMyDataSpaceGreaterThan(DataSpace<dim> other)
        {
            return !other.isOneDimensionGreaterThan(data_space);
        }

        DataSpace<dim> data_space;
        DataSpace<dim> m_physicalMemorySize;

        size_t *current_size;

        bool data1D;
    };

} // namespace mem

} // namespace pmacc

namespace redGrapes
{
namespace trait
{

template< typename Buffer >
TRAIT_BUILD_RESOURCE_PROPERTIES( pmacc::mem::buffer::size::ReadGuard<Buffer> );
template< typename Buffer >
TRAIT_BUILD_RESOURCE_PROPERTIES( pmacc::mem::buffer::size::WriteGuard<Buffer> );
template< typename Buffer >
TRAIT_BUILD_RESOURCE_PROPERTIES( pmacc::mem::buffer::data::ReadGuard<Buffer> );
template< typename Buffer >
TRAIT_BUILD_RESOURCE_PROPERTIES( pmacc::mem::buffer::data::WriteGuard<Buffer> );
template< typename Buffer >
TRAIT_BUILD_RESOURCE_PROPERTIES( pmacc::mem::buffer::ReadGuard<Buffer> );
template< typename Buffer >
TRAIT_BUILD_RESOURCE_PROPERTIES( pmacc::mem::buffer::WriteGuard<Buffer> );

} // namespace trait
}


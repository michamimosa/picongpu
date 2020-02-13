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

template < typename Buffer >
struct ReadGuard;
template < typename Buffer >
struct WriteGuard;

namespace size
{
    template < typename Buffer >
    struct ReadGuard : std::shared_ptr< Buffer >
    {
        operator rg::ResourceAccess() const
        {
            return this->std::shared_ptr<Buffer>::get()->read_size();
        }

        ReadGuard read() const noexcept { return *this; }
        std::size_t get() const { return this->std::shared_ptr<Buffer>::get()->get_size(); }
        auto data_space() const { return this->std::shared_ptr<Buffer>::get()->getCurrentDataSpace(); }

    protected:
        friend class ::pmacc::mem::buffer::ReadGuard< Buffer >;
        friend Buffer;

        ReadGuard( std::shared_ptr< Buffer > obj )
            : std::shared_ptr< Buffer >( obj )
        {}        
    };

    template < typename Buffer >
    struct WriteGuard : ReadGuard< Buffer >
    {
        operator rg::ResourceAccess() const
        {
            return this->std::shared_ptr<Buffer>::get()->write_size();
        }

        WriteGuard write() { return *this; }
        void set( size_t const new_size ) const { this->std::shared_ptr<Buffer>::get()->set_size( new_size ); }

    protected:
        friend class ::pmacc::mem::buffer::WriteGuard< Buffer >;
        friend Buffer;

        WriteGuard( std::shared_ptr< Buffer > obj )
            : ReadGuard< Buffer >( obj )
        {}
    };
} // namespace size

namespace data
{
    template < typename Buffer >
    struct ReadGuard : std::shared_ptr< Buffer >
    {
        operator rg::ResourceAccess() const
        {
            return this->get()->access_data( rg::access::IOAccess::read );
        }

        ReadGuard read() const noexcept { return *this; }

        using Item = typename Buffer::Item;
        using DataBox = typename Buffer::DataBoxType;

        bool is1D() const { return this->get()->is1D(); }
        Item const * getPointer() const { return this->get()->getPointer(); }
        Item const * getBasePointer() const { return this->get()->getBasePointer(); }
        DataBox const getDataBox() const { return this->get()->getDataBox(); }

    protected:
        friend class ::pmacc::mem::buffer::ReadGuard< Buffer >;
        friend Buffer;

        ReadGuard( std::shared_ptr< Buffer > obj )
            : std::shared_ptr< Buffer >( obj )
        {}
    };

    template < typename Buffer >
    struct WriteGuard : ReadGuard< Buffer >
    {
        operator rg::ResourceAccess() const
        {
            return this->get()->access_data( rg::access::IOAccess::write );
        }

        WriteGuard write() const noexcept { return *this; }

        using Item = typename Buffer::Item;
        using DataBox = typename Buffer::DataBoxType;

        void reset() const { this->get()->reset(); }
        void fill(Item const & item) const { this->get()->fill(item); }
        Item * getPointer() const { return this->get()->getPointer();  }
        Item * getBasePointer() const { return this->get()->getBasePointer(); }
        DataBox getDataBox() const { return this->get()->getDataBox(); }

    protected:
        friend class ::pmacc::mem::buffer::WriteGuard< Buffer >;
        friend Buffer;

        WriteGuard( std::shared_ptr< Buffer > obj )
            : ReadGuard< Buffer >( obj )
        {}
    };
} // namespace data

template < typename Buffer >
struct ReadGuard : public std::shared_ptr< Buffer >
{
    ReadGuard read() const noexcept { return *this; }

    auto data() const noexcept { return data::ReadGuard<Buffer>( *this ); }
    auto size() const noexcept { return size::ReadGuard<Buffer>( *this ); }

protected:
    friend Buffer;

    ReadGuard( std::shared_ptr< Buffer > obj )
        : std::shared_ptr< Buffer >( obj )
    {}
};

template < typename Buffer >
struct WriteGuard : ReadGuard< Buffer >
{
    WriteGuard write() const noexcept { return *this; }

    auto data() const noexcept { return data::WriteGuard<Buffer>( (std::shared_ptr<Buffer>)*this ); }
    auto size() const noexcept { return size::WriteGuard<Buffer>( (std::shared_ptr<Buffer>)*this ); }

protected:
    friend Buffer;

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
        : buffer::WriteGuard<Buffer>( std::make_shared<Buffer>() )
    {
        this->get()->init( std::forward<Args>(args)... );
    }

    BufferResource( std::shared_ptr<Buffer> obj )
        : buffer::WriteGuard<Buffer>( obj )
    {}
};

    /**
     * Minimal function description of a buffer,
     *
     * @tparam TYPE data type stored in the buffer
     * @tparam DIM dimension of the buffer (1-3)
     */
    template <
        typename T_Item,
        std::size_t T_dim,
        typename T_DataAccessPolicy = rg::access::IOAccess
    >
    class Buffer
        : public std::enable_shared_from_this< Buffer<T_Item, T_dim, T_DataAccessPolicy> >
    {
    protected:
        template < typename Derived >
        std::shared_ptr< Derived > shared_from_base()
        {
            return std::static_pointer_cast< Derived >( this->shared_from_this() );
        }

        auto buffer_resource()
        {
            return BufferResource<Buffer<T_Item, dim, T_DataAccessPolicy>>(this->shared_from_this());
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
        void init(DataSpace< dim > size,
                  DataSpace< dim > physicalMemorySize)
        {
            Environment<>::task(
                [obj=this->shared_from_this(), size, physicalMemorySize] {
                    obj->data_space = DataSpace< dim >( size );
                    obj->data1D = true;
                    obj->m_physicalMemorySize = physicalMemorySize;

                    CUDA_CHECK(cudaMallocHost((void**)&obj->current_size, sizeof(size_t)));
                    *obj->current_size = size.productOfComponents();
                },
                TaskProperties::Builder()
                    .label("Buffer::Buffer()")
                    .resources({ this->write_size() }));
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
        virtual Item * getBasePointer() const = 0;

        /*! Get pointer that includes all offsets
         * @return pointer to a point in a memory array
         */
        virtual Item * getPointer() const = 0;

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
                    .resources({ this->read_size() })
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
                    .resources({ this->write_size() })
            );
        }

        virtual void reset( bool preserveData = false ) = 0;
        virtual void fill( Item const & value ) = 0;

        virtual DataBox< PitchedBox<Item, dim> > getDataBox() = 0;

        inline bool is1D() const noexcept
        {
            return data1D;
        }

        rg::ResourceAccess access_size( rg::access::IOAccess mode ) const
        {
            return this->resource_size.make_access( mode );
        }

        rg::ResourceAccess read_size() const
        {
            return this->access_size( rg::access::IOAccess::read );
        }

        rg::ResourceAccess write_size() const
        {
            return this->access_size( rg::access::IOAccess::write );
        }

        rg::ResourceAccess access_data( T_DataAccessPolicy acc ) const
        {
            return this->resource_data.make_access( acc );
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

        rg::Resource< rg::access::IOAccess > resource_size;
        rg::Resource< T_DataAccessPolicy > resource_data;

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

template < typename Buffer >
struct BuildProperties< pmacc::mem::buffer::ReadGuard<Buffer> >
{
    template < typename Builder >
    static void build(Builder & builder, pmacc::mem::buffer::ReadGuard<Buffer> const & buf)
    {
        builder.add( buf.size() );
        builder.add( buf.data() );
    }
};

template < typename Buffer >
struct BuildProperties< pmacc::mem::buffer::WriteGuard<Buffer> >
{
    template < typename Builder >
    static void build(Builder & builder, pmacc::mem::buffer::WriteGuard<Buffer> const & buf)
    {
        builder.add( buf.size() );
        builder.add( buf.data() );
    }
};

} // namespace trait

} // namespace pmacc


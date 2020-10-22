
#pragma once

#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>

#include <redGrapes/resource/resource.hpp>

namespace rg = redGrapes;

namespace pmacc
{
namespace mem
{
namespace buffer
{

/*!
 * Base class for accessing a buffer (both host and device).
 * It defines a subdomain that is accessed with
 * the vectors offset and data_space.
 *
 * @tparam Buffer must satisfy the Buffer concept
 */
template < typename Buffer >
struct GuardBase
{
    using Item = typename Buffer::Item;
    static constexpr std::size_t dim = Buffer::dim;
    using DataBoxType = DataBox< PitchedBox< Item, dim > >;

    template < typename, typename >
    friend class rg::trait::BuildProperties;

    /*!
     * @return if this buffer is bytewise copyable,
     *         regardless of dataspace
     *
     * true for all views that span over the complete
     * data instead of a subdomain
     */
    bool is1D() const noexcept
    {
        return data1D;
    }

    /*! Get max spread (elements) of any dimension
     * @return spread (elements) per dimension
     */    
    DataSpace< dim > getDataSpace() const noexcept
    {
        return size.get_data_space();
    }

    /*!
     */
    DataSpace< dim > getOffset() const noexcept
    {
        return offset;
    }

    /*! get pitch of the base buffer
     */
    std::size_t getPitch() const noexcept
    {
        return data.obj->getPitch();
    }

    //protected:
    friend Buffer;

    /*! Create a subdomain
     */
    GuardBase(
        rg::SharedResourceObject<
            typename Buffer::Data,
            typename Buffer::DataAccessPolicy
        > const & data,
        DataSpace< dim > offset,
        DataSpace< dim > data_space
    ) :
        data( data ),
        offset( offset ),
        data1D( false )
    {
        size.reset( data_space );
    }

    /*! Create a complete view of the underlying data
     */
    GuardBase(
        rg::SharedResourceObject<
            typename Buffer::Data,
            typename Buffer::DataAccessPolicy
        > const & data
    ) :
        GuardBase(
            data,
            DataSpace< dim >(),
            data.obj->get_capacity()
        )
    {
        data1D = true;
    }

    GuardBase(
        rg::SharedResourceObject<
            typename Buffer::Data,
            typename Buffer::DataAccessPolicy
        > const & data,
        GuardBase const & other
    ) :
        data( data ),
        offset( other.offset ),
        size( other.size ),
        data1D( other.data1D )
    {}

    GuardBase( GuardBase const & other )
        : data( other.data ),
          offset( other.offset ),
          size( other.size ),
          data1D( other.data1D )
    {}

    GuardBase sub_area(
        DataSpace< Buffer::dim > offset,
        DataSpace< Buffer::dim > data_space
    )
    {
        return GuardBase( this->data, this->offset + offset, data_space );
    }

    DataBoxType get_data_box( DataSpace< dim > offset ) const noexcept
    {
        this->obj->get_data_box( offset );
    }

    rg::SharedResourceObject< typename Buffer::Data, typename Buffer::DataAccessPolicy > data;

    DataSpace< dim > offset;
    typename Buffer::Size size;
    bool data1D;
};


/*
 * access-guards for size
 */

namespace size
{

template < typename Buffer >
struct ReadGuard : GuardBase< Buffer >
{
    auto read() { return *this; }

    std::size_t get() const { return this->size.get_current_size(); }
    std::size_t getCurrentSize() const { return this->size.get_current_size(); }
    DataSpace< Buffer::dim > getCurrentDataSpace() const { return this->size.get_current_data_space(); }

    ReadGuard( GuardBase< Buffer > const & base )
        : GuardBase< Buffer >( base ) {}
};

template < typename Buffer >
struct WriteGuard : ReadGuard< Buffer >
{
    auto write() { return *this; }

    void reset() { this->size.reset(); }
    void set( std::size_t new_size ) { this->size.set_current_size( new_size ); }
    void setCurrentSize( std::size_t new_size ) { this->size.set_current_size( new_size ); }

    WriteGuard( GuardBase< Buffer > const & base )
        : ReadGuard< Buffer >( base ) {}
};

} // namespace size




/*
 * access-guards for data
 */

namespace data
{

template < typename Buffer >
struct ReadGuardBase : GuardBase< Buffer >
{
    using Item = typename Buffer::Item;
    using DataBox = typename Buffer::DataBoxType;

    Item const * getBasePointer() const { return this->data.obj->get_base_ptr(); }
    Item const * getPointer() const { return this->data.obj->get_pointer( this->offset ); }
    DataBox const getDataBox() const { return this->data.obj->get_data_box( this->offset ); }

    ReadGuardBase( GuardBase< Buffer > const & base ) noexcept
        : GuardBase< Buffer >( base ) {}
};

template < typename Buffer >
struct WriteGuardBase : ReadGuardBase< Buffer >
{
    using Item = typename Buffer::Item;
    using DataBox = typename Buffer::DataBoxType;

    Item * getBasePointer() const { return this->data.obj->get_base_ptr(); }
    Item * getPointer() const { return this->data.obj->get_pointer( this->offset ); }
    DataBox getDataBox() const { return this->data.obj->get_data_box( this->offset ); }

    WriteGuardBase( GuardBase< Buffer > const & base ) noexcept
        : ReadGuardBase< Buffer >( base ) {}
};


template <
    typename Buffer,
    typename Sfinae = void
>
struct ReadGuard
    : ReadGuardBase< Buffer >
{
    ReadGuard( GuardBase< Buffer > const & base ) noexcept
        : ReadGuardBase< Buffer >( base ) {}

    ReadGuard read() const { return *this; }
};

template <
    typename Buffer,
    typename Sfinae = void
>
struct WriteGuard
    : WriteGuardBase< Buffer >
{
    WriteGuard( GuardBase< Buffer > const & base ) noexcept
        : WriteGuardBase< Buffer > ( base ) {}

    ReadGuard< Buffer > read() const { return *this; }
    WriteGuard write() const  { return *this; }
};

} // namespace data




/*
 * access-guards for buffers
 */

template < typename Buffer >
struct WriteGuard;

template < typename Buffer >
struct ReadGuard
    : GuardBase< Buffer >
{
    auto size() const noexcept { return typename Buffer::SizeGuard( *this ).read(); }
    auto data() const noexcept { return typename Buffer::DataGuard( *this ).read(); }

    ReadGuard< Buffer > read() const noexcept { return *this; }

    ReadGuard< Buffer > sub_area( DataSpace< Buffer::dim > offset, DataSpace< Buffer::dim > data_space )
    {
        return ReadGuard(this->GuardBase<Buffer>::sub_area(offset, data_space));
    }

    ReadGuard( GuardBase< Buffer > const & base )
        : GuardBase< Buffer >( base )
    {}
};

template < typename Buffer >
struct WriteGuard
    : ReadGuard< Buffer >
{
    auto size() const noexcept { return typename Buffer::SizeGuard( *this ).write(); }
    auto data() const noexcept { return typename Buffer::DataGuard( *this ).write(); }

    WriteGuard< Buffer > write() const noexcept { return *this; }

    WriteGuard< Buffer > sub_area( DataSpace< Buffer::dim > offset, DataSpace< Buffer::dim > data_space )
    {
        return WriteGuard(this->GuardBase<Buffer>::sub_area(offset, data_space));
    }

    WriteGuard( GuardBase< Buffer > const & base )
        : ReadGuard< Buffer >( base )
    {}
}; 

} // namespace buffer

} // namespace mem

} // namespace pmacc


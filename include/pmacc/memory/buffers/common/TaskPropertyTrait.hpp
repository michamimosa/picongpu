
#pragma once

#include <redGrapes/property/trait.hpp>
#include <pmacc/memory/buffers/common/Resource.hpp>
#include <pmacc/memory/buffers/common/Size.hpp>
#include <pmacc/memory/buffers/common/Data.hpp>

namespace redGrapes
{
namespace trait
{

/*
 * task-property builders for buffer acccess-guards
 */

template < typename Buffer >
struct BuildProperties<
    pmacc::mem::buffer::ReadGuard< Buffer >
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::buffer::ReadGuard< Buffer > const & buf
    )
    {
        builder.add( buf.size() );
        builder.add( buf.data() );
    }
};

template < typename Buffer >
struct BuildProperties<
    pmacc::mem::buffer::WriteGuard< Buffer >
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::buffer::WriteGuard< Buffer > const & buf
    )
    {
        builder.add( buf.size() );
        builder.add( buf.data() );
    }
};



/*
 * task-property builders for size access-guards that use the common BufferSize (size stored on host only)
 */

template < typename Buffer >
struct BuildProperties<
    pmacc::mem::buffer::size::ReadGuard< Buffer >,
    typename std::enable_if<
        std::is_same<
            typename Buffer::Size,
            pmacc::mem::buffer::BufferSize< Buffer::dim >
        >::value
    >::type
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::buffer::size::ReadGuard< Buffer > const & buf
    )
    {
        builder.add( buf.size.host_current_size.read() );
    }
};

template < typename Buffer >
struct BuildProperties<
    pmacc::mem::buffer::size::WriteGuard< Buffer >,
    typename std::enable_if<
        std::is_same<
            typename Buffer::Size,
            pmacc::mem::buffer::BufferSize< Buffer::dim >
        >::value
    >::type
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::buffer::size::WriteGuard< Buffer > const & buf
    )
    {
        builder.add( buf.size.host_current_size.write() );
    }
};



/*
 * task-property builders for data access-guards that use an IOAccess policy
 */

template < typename Buffer >
struct BuildProperties<
    pmacc::mem::buffer::data::ReadGuard< Buffer >,
    typename std::enable_if<
        std::is_same<
            typename Buffer::DataAccessPolicy,
            rg::access::IOAccess
        >::value
    >::type
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::buffer::data::ReadGuard< Buffer > const & buf
    )
    {
        builder.add( buf.data.make_access( rg::access::IOAccess::write ) );
    }
};

template < typename Buffer >
struct BuildProperties<
    pmacc::mem::buffer::data::WriteGuard< Buffer >,
    typename std::enable_if<
        std::is_same<
            typename Buffer::DataAccessPolicy,
            rg::access::IOAccess
        >::value
    >::type
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::buffer::data::WriteGuard< Buffer > const & buf
    )
    {
        builder.add( buf.data.make_access( rg::access::IOAccess::write ) );
    }
};

} // namespace trait

} // namespace redGrapes


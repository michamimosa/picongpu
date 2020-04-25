/* Copyright 2020 Michael Sippel
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

#include <redGrapes/access/io.hpp>

namespace pmacc
{
namespace mem
{
namespace grid_buffer
{
namespace data
{

/*!
 * AccessPolicy for GridBuffers.
 * Used by redGrapes to calculate task dependencies
 */
struct Access
{
    //! how is the data used (read or write?)
    rg::access::IOAccess mode;

    /*! which areas area accessed?
     * (e.g. CORE, CORE+BORDER)
     */
    uint32_t area;

    /*! which directions are used?
     * this applies only for BORDER and GUARD.
     */
    Mask direction;

    /*
    //! really neccessary?, will be initialized in ExchangeGuard
    Access(
        rg::access::IOAccess mode = rg::access::IOAccess::write,
        uint32_t area = CORE + BORDER + GUARD,
        Mask direction =
            Mask(TOP) + Mask(BOTTOM) +
            Mask(LEFT) + Mask(RIGHT) +
            Mask(FRONT) + Mask(BACK)
    ) :
        mode( mode ),
        area( area ),
        direction( direction )
    {}
    */

    /*! @return whether these accesses must be sequential and in-order.
     */
    static bool is_serial(
        Access const & a,
        Access const & b
    )
    {
        if(
            // overlapping area ?
            // (e.g. CORE+BORDER+GUARD & GUARD )
            ( a.area & b.area )
            &&
            // two reads area always parallel,
            // regardless of direction or area
            rg::access::IOAccess::is_serial(a.mode, b.mode)
        )
        {
            // core is unrelated to direction
            if(
                ( a.area & CORE ) &&
                ( b.area & CORE )
            )
                return true;

            // test directions for border & guard
            else
            {
                for( int ex = 0; ex < 27; ++ex )
                    if(
                        a.direction.containsExchangeType(ex) &&
                        b.direction.containsExchangeType(ex)
                    )
                        // found one direction that is used by both
                        return true;

                return false;
            }
        }
        else
            return false;
    }

    bool is_superset_of( Access const & other ) const
    {
        if(
            // other doesn't have any area thats not included in this->area
            (~this->area & other.area) == 0
            &&
            this->mode.is_superset_of( other.mode )
        )
        {
            // other is core only, we don't need to check directions
            if( other.area == CORE )
                return true;
            else
            {
                for( uint32_t ex = 0; ex < 27; ++ex )
                    if(
                        ! this->direction.containsExchangeType(ex) &&
                        other.direction.containsExchangeType(ex)
                    )
                        // found a direction that is used in other,
                        // but not in this
                        return false;

                return true;
            }
        }
        else
            return false;
    }

    friend bool operator== (
        Access const & a,
        Access const & b
    )
    {
        return
            a.area == b.area &&
            a.direction == b.direction;
    }

    friend std::ostream & operator<<(std::ostream & out, Access const & a)
    {
        pmacc::type::ExchangeTypeNames names;

        out << "GridAccess { mode = " << a.mode << ", area = {";
        if( a.area & CORE )
            out << "CORE;";
        if( a.area & BORDER )
            out << "BORDER;";
        if( a.area & GUARD )
            out << "GUARD;";
        out << "}, direction = " << names[a.direction] << " }";

        return out;
    }
};

} // namespace data

} // namespace grid_buffer



namespace buffer
{
namespace data
{

template < typename Buffer >
struct ReadGuard<
    Buffer,
    typename std::enable_if<
        std::is_same<
            typename Buffer::DataAccessPolicy,
            pmacc::mem::grid_buffer::data::Access
        >::value
    >::type
>
    : ReadGuardBase< Buffer >
{
    uint32_t area;
    Mask directions;

    friend Buffer;
    friend class rg::trait::BuildProperties< ReadGuard >;

public:
    ReadGuard( GuardBase< Buffer > const & base )
        : ReadGuardBase< Buffer >( base )
        , area{ CORE + BORDER + GUARD }
        , directions(
            Mask(TOP) + Mask(BOTTOM) +
            Mask(LEFT) + Mask(RIGHT) +
            Mask(FRONT) + Mask(BACK))
    {}

    ReadGuard read() const { return *this; }
};

template < typename Buffer >
struct WriteGuard<
    Buffer,
    typename std::enable_if<
        std::is_same<
            typename Buffer::DataAccessPolicy,
            pmacc::mem::grid_buffer::data::Access
        >::value
    >::type
>
    : WriteGuardBase< Buffer >
{
public:
    uint32_t area;
    Mask directions;

    friend Buffer;
    friend class rg::trait::BuildProperties< WriteGuard >;

    WriteGuard( GuardBase< Buffer > const & base )
        : WriteGuardBase< Buffer >( base )
        , area{ CORE + BORDER + GUARD }
        , directions(
            Mask(TOP) + Mask(BOTTOM) +
            Mask(LEFT) + Mask(RIGHT) +
            Mask(FRONT) + Mask(BACK))
    {}

    ReadGuard< Buffer > read() const { return *this; }
    WriteGuard write() const { return *this; }
};

} // namespace data
} // namespace buffer

} // namespace mem

} // namespace pmacc



namespace redGrapes
{
namespace trait
{

template < typename Buffer >
struct BuildProperties<
    pmacc::mem::buffer::data::ReadGuard< Buffer >,
    typename std::enable_if<
        std::is_same<
            typename Buffer::DataAccessPolicy,
            pmacc::mem::grid_buffer::data::Access
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
        builder.add(
            buf.data.make_access(
                pmacc::mem::grid_buffer::data::Access{
                    rg::access::IOAccess::read,
                    buf.area,
                    buf.directions
                }));
    }
};

template < typename Buffer >
struct BuildProperties<
    pmacc::mem::buffer::data::WriteGuard< Buffer >,
    typename std::enable_if<
        std::is_same<
            typename Buffer::DataAccessPolicy,
            pmacc::mem::grid_buffer::data::Access
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
        builder.add(
            buf.data.make_access(
                 pmacc::mem::grid_buffer::data::Access{
                    rg::access::IOAccess::write,
                    buf.area,
                    buf.directions
                }));
    }
};

} // namespace trait

} // namespace redGrapes



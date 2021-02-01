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
#include <pmacc/memory/buffers/common/Resource.hpp>
#include <pmacc/memory/buffers/gridBuffer/Exchange.hpp>

#include <sstream>
#include <fmt/format.h>

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
            // two reads are always parallel,
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
                for( uint32_t ex = 1; ex < 27; ++ex )
                    if(
                        ! this->direction.isSet(ex) &&
                        other.direction.isSet(ex)
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
};

} // namespace data

} // namespace grid_buffer

template < typename Buffer >
struct buffer::data::ReadGuard<
    Buffer,
    typename std::enable_if<
        std::is_same<
            typename Buffer::DataAccessPolicy,
            pmacc::mem::grid_buffer::data::Access
        >::value
    >::type
>
    : buffer::data::ReadGuardBase< Buffer >
{
    uint32_t area;
    Mask directions;

    friend Buffer;
    friend class rg::trait::BuildProperties< ReadGuard >;
    
    ReadGuard( ReadGuard const & other )
        : buffer::data::ReadGuardBase< Buffer >( other )
        , area(other.area)
        , directions(other.directions)
    {}

    ReadGuard( buffer::data::WriteGuard< Buffer > const & other )
        : buffer::data::ReadGuardBase< Buffer >( other )
        , area(other.area)
        , directions(other.directions)
    {}

    //! read guard for whole buffer
    ReadGuard( GuardBase< Buffer > const & base )
        : buffer::data::ReadGuardBase< Buffer >( base )
        , area{ CORE + BORDER + GUARD }
        , directions(
            Mask(TOP) + Mask(BOTTOM) +
            Mask(LEFT) + Mask(RIGHT) +
            Mask(FRONT) + Mask(BACK))
    {}

    ReadGuard( GuardBase< Buffer > const & base, uint32_t const & area, Mask const & directions )
        : buffer::data::ReadGuardBase< Buffer >( base )
        , area(area)
        , directions(directions)
    {}

    //! create read guard for exchange data
    ReadGuard(
        GuardBase< Buffer > const & base,
        ExchangeType exchangeType,
        AreaType dataPlace,
        DataSpace< Buffer::dim > guardingCells,
        GridLayout< Buffer::dim > gridLayout
    )
        : buffer::data::ReadGuardBase< Buffer >(
              base.sub_area(
                  exchange::exchangeTypeToOffset< Buffer::dim >(
                      exchangeType,
                      gridLayout,
                      guardingCells,
                      dataPlace == GUARD ? GUARD : BORDER
                  ),
                  exchange::exchangeTypeToDataSpace< Buffer::dim >(
                      exchangeType,
                      gridLayout,
                      guardingCells
                  )
              )
          )
        , directions( Mask(exchangeType) )
        , area( dataPlace )
    {}

    std::vector< rg::ResourceAccess > get_access() const
    {
        return
            std::vector< rg::ResourceAccess >({
                this->data.make_access(
                    grid_buffer::data::Access{
                        rg::access::IOAccess::read,
                        this->area,
                        this->directions
                    }
                )   
            });
    }

    auto read() const noexcept
    {
        return ReadGuard( *this, this->area, this->directions );
    }

    //! only reduces resource access, not memory offset
    auto access_dataPlace( uint32_t area ) const
    {
        auto n = typename Buffer::DataGuard( *this ).read();
        n.area = area;
        n.directions = this->directions;
        return n;
    }

    //! only reduces resource access, not memory offset
    auto access_directions( Mask directions ) const
    {
        auto n = typename Buffer::DataGuard( *this ).read();
        n.area = this->area;
        n.directions = directions;
        return n;
    }

    auto exchange(
        ExchangeType exchangeType,
        AreaType dataPlace,
        DataSpace< Buffer::dim > guardingCells,
        GridLayout< Buffer::dim > gridLayout
    )
    {
        return ReadGuard( *this, exchangeType, dataPlace, guardingCells, gridLayout );
    }
};

template < typename Buffer >
struct buffer::data::WriteGuard<
    Buffer,
    typename std::enable_if<
        std::is_same<
            typename Buffer::DataAccessPolicy,
            pmacc::mem::grid_buffer::data::Access
        >::value
    >::type
>
    : buffer::data::WriteGuardBase< Buffer >
{
public:
    uint32_t area;
    Mask directions;

    friend Buffer;
    friend class rg::trait::BuildProperties< WriteGuard >;

    WriteGuard( WriteGuard const & other )
        : buffer::data::WriteGuardBase< Buffer >( other )
        , area(other.area)
        , directions(other.directions)
    {}

    WriteGuard( GuardBase< Buffer > const & base )
        : buffer::data::WriteGuardBase< Buffer >( base )
        , area{ CORE + BORDER + GUARD }
        , directions(
            Mask(TOP) + Mask(BOTTOM) +
            Mask(LEFT) + Mask(RIGHT) +
            Mask(FRONT) + Mask(BACK))
    {}

    WriteGuard( GuardBase< Buffer > const & base, uint32_t const & area, Mask const & directions )
        : buffer::data::WriteGuardBase< Buffer >( base )
        , area(area)
        , directions(directions)
    {}

    //! create write guard for exchange data
    WriteGuard(
        GuardBase< Buffer > const & base,
        ExchangeType exchangeType,
        AreaType dataPlace,
        DataSpace< Buffer::dim > guardingCells,
        GridLayout< Buffer::dim > gridLayout
    )
        : buffer::data::WriteGuardBase< Buffer >(
              base.sub_area(
                  exchange::exchangeTypeToOffset< Buffer::dim >(
                      exchangeType,
                      gridLayout,
                      guardingCells,
                      dataPlace == GUARD ? GUARD : BORDER
                  ),
                  exchange::exchangeTypeToDataSpace< Buffer::dim >(
                      exchangeType,
                      gridLayout,
                      guardingCells
                  )
              )
          )
        , directions( Mask(exchangeType) )
        , area( dataPlace )
    {}

    std::vector< rg::ResourceAccess > get_access() const
    {
        return
            std::vector< rg::ResourceAccess >({
                this->data.make_access(
                    grid_buffer::data::Access{
                        rg::access::IOAccess::write,
                        this->area,
                        this->directions
                    }
                )   
            });
    }

    auto read() const noexcept
    {
        return buffer::data::ReadGuard< Buffer >( *this, this->area, this->directions );
    }

    auto write() const noexcept
    {
        return WriteGuard( *this, this->area, this->directions );
    }

    //! only reduces resource access, not memory offset
    auto access_dataPlace( uint32_t area ) const
    {
        // todo: assert area is superset of this->area

        typename Buffer::DataGuard n( *this );
        n.area = area;
        n.directions = this->directions;
        return n;
    }

    //! only reduces resource access, not memory offset
    auto access_directions( Mask directions ) const
    {
        // todo: assert directions is superset of this->directions

        typename Buffer::DataGuard n( *this );
        n.area = this->area;
        n.directions = directions;
        return n;
    }

    auto exchange(
        ExchangeType exchangeType,
        AreaType dataPlace,
        DataSpace< Buffer::dim > guardingCells,
        GridLayout< Buffer::dim > gridLayout
    )
    {
        return buffer::data::WriteGuard< Buffer >( *this, exchangeType, dataPlace, guardingCells, gridLayout );
    }    
};

template < typename Buffer >
struct buffer::ReadGuard<
    Buffer,
    typename std::enable_if<
        std::is_same<
            typename Buffer::DataAccessPolicy,
            pmacc::mem::grid_buffer::data::Access
        >::value
    >::type
>
    : buffer::GuardBase< Buffer >
{
    uint32_t area;
    Mask directions;

    friend Buffer;
    friend class rg::trait::BuildProperties< ReadGuard >;

public:
    ReadGuard( GuardBase< Buffer > const & base )
        : buffer::GuardBase< Buffer >( base )
        , area{ CORE + BORDER + GUARD }
        , directions(
            Mask(TOP) + Mask(BOTTOM) +
            Mask(LEFT) + Mask(RIGHT) +
            Mask(FRONT) + Mask(BACK))
    {}

    ReadGuard( GuardBase< Buffer > const & base, uint32_t const & area, Mask const & directions )
        : buffer::GuardBase< Buffer >( base )
        , area(area)
        , directions(directions)
    {}
    
    ReadGuard(
        GuardBase< Buffer > const & base,
        ExchangeType exchangeType,
        AreaType dataPlace,
        DataSpace< Buffer::dim > guardingCells,
        GridLayout< Buffer::dim > gridLayout
    )
        : buffer::GuardBase< Buffer >(
              base.sub_area(
                  exchange::exchangeTypeToOffset< Buffer::dim >(
                      exchangeType,
                      gridLayout,
                      guardingCells,
                      dataPlace == GUARD ? GUARD : BORDER
                  ),
                  exchange::exchangeTypeToDataSpace< Buffer::dim >(
                      exchangeType,
                      gridLayout,
                      guardingCells
                  )
              )
          )
        , directions( Mask(exchangeType) )
        , area( dataPlace )
    {}

    std::vector< rg::ResourceAccess > get_access() const
    {
        std::vector< rg::ResourceAccess > acc;

        auto size_acc = this->size().get_access();
        acc.insert( std::begin(acc), std::begin(size_acc), std::end(size_acc) );

        auto data_acc = this->data().get_access();
        acc.insert( std::begin(acc), std::begin(data_acc), std::end(data_acc) );

        return acc;
    }
    
    auto size() const noexcept { return typename Buffer::SizeGuard( *this ).read(); }

    auto data() const noexcept
    {
        return typename Buffer::DataGuard( *this )
            .read()
            .access_dataPlace( this->area )
            .access_directions( this->directions );
    }

    auto read() const { return ReadGuard( *this, this->area, this->directions ); }

    auto exchange(
        ExchangeType exchangeType,
        AreaType dataPlace,
        DataSpace< Buffer::dim > guardingCells,
        GridLayout< Buffer::dim > gridLayout
    )
    {
        return ReadGuard( *this, exchangeType, dataPlace, guardingCells, gridLayout );
    }
};

template < typename Buffer >
struct buffer::WriteGuard<
    Buffer,
    typename std::enable_if<
        std::is_same<
            typename Buffer::DataAccessPolicy,
            pmacc::mem::grid_buffer::data::Access
        >::value
    >::type
>
    : buffer::ReadGuard< Buffer >
{
public:
    friend Buffer;
    friend class rg::trait::BuildProperties< WriteGuard >;

    WriteGuard( GuardBase< Buffer > const & base )
        : buffer::ReadGuard< Buffer >( base )
    {}

    WriteGuard( GuardBase< Buffer > const & base, uint32_t const & area, Mask const & directions )
        : buffer::ReadGuard< Buffer >( base, area, directions )
    {}

    WriteGuard(
        GuardBase< Buffer > const & base,
        ExchangeType exchangeType,
        AreaType dataPlace,
        DataSpace< Buffer::dim > guardingCells,
        GridLayout< Buffer::dim > gridLayout
    )
        : buffer::ReadGuard< Buffer >( base, exchangeType, dataPlace, guardingCells, gridLayout )
    {}

    std::vector< rg::ResourceAccess > get_access() const
    {
        std::vector< rg::ResourceAccess > acc;

        auto size_acc = this->size().get_access();
        acc.insert( std::begin(acc), std::begin(size_acc), std::end(size_acc) );

        auto data_acc = this->data().get_access();
        acc.insert( std::begin(acc), std::begin(data_acc), std::end(data_acc) );

        return acc;
    }

    auto size() const noexcept { return typename Buffer::SizeGuard( *this ).write(); }

    auto data() const noexcept
    {
        return typename Buffer::DataGuard( *this )
            .write()
            .access_dataPlace( this->area )
            .access_directions( this->directions );
    }

    auto write() const noexcept { return WriteGuard( *this, this->area, this->directions ); }

    auto exchange(
        ExchangeType exchangeType,
        AreaType dataPlace,
        DataSpace< Buffer::dim > guardingCells,
        GridLayout< Buffer::dim > gridLayout
    )
    {
        return WriteGuard( *this, exchangeType, dataPlace, guardingCells, gridLayout );
    }
};

} // namespace mem

} // namespace pmacc


template <>
struct fmt::formatter< pmacc::mem::grid_buffer::data::Access >
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        pmacc::mem::grid_buffer::data::Access const & a,
        FormatContext & ctx
    )
    {
        pmacc::type::ExchangeTypeNames names;

        bool first = true;
        std::stringstream area_str;
        area_str << "[ ";

        if( a.area & pmacc::CORE )
        {
            first = false;
            area_str << " \"Core\"";
        }

        if( a.area & pmacc::BORDER )
        {
            if( ! first)
                area_str << ", ";

            first = false;
            area_str << "\"Border\"";
        }

        if( a.area & pmacc::GUARD )
        {
            if( ! first)
                area_str << ", ";

            first = false;
            area_str << "\"Guard\"";
        }

        area_str << " ]";


        std::stringstream direction_str;
        direction_str << "[";

        first = true;
        for(int ex=1; ex<27; ++ex)
            if( a.direction.isSet(ex) )
            {
                if(! first)
                    direction_str << ", ";

                first = false;
                direction_str << "\"" << names[ex] << "\"";
            }

        direction_str << "]";

        return format_to(
                   ctx.out(),
                   "{{ \"GridAccess\" : {{ \"mode\" : {}, \"area\" : {}, \"directions\" : {} }} }}",
                   a.mode,
                   area_str.str(),
                   direction_str.str()
               );
    }
};




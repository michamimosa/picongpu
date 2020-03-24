/* Copyright 2013-2019 Rene Widera, Benjamin Worpitz, Alexander Grund
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

#include "pmacc/dimensions/GridLayout.hpp"
#include "pmacc/mappings/simulation/EnvironmentController.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/memory/buffers/ExchangeIntern.hpp"
#include <pmacc/memory/buffers/Buffer.hpp>
#include <pmacc/memory/buffers/HostBufferIntern.hpp>
#include <pmacc/memory/buffers/DeviceBufferIntern.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <set>

namespace pmacc
{
namespace mem
{
namespace privateGridBuffer
{

class UniquTag
{
public:

    static UniquTag& getInstance()
    {
        static UniquTag instance;
        return instance;
    }

    bool isTagUniqu(uint32_t tag)
    {
        bool isUniqu = tags.find(tag) == tags.end();
        if (isUniqu)
            tags.insert(tag);
        return isUniqu;
    }
private:

    UniquTag()
    {
    }

    /**
     * Constructor
     */
    UniquTag(const UniquTag&)
    {

    }

    std::set<uint32_t> tags;
};

}//end namespace privateGridBuffer

namespace grid_buffer
{
namespace data
{
struct Access
{
    rg::access::IOAccess mode;

    uint32_t area;
    uint32_t direction;

    static bool is_serial( Access const & a, Access const & b )
    {
        if(
            a.area == b.area &&
            rg::access::IOAccess::is_serial(a.mode, b.mode)
        )
        {
            switch( a.area )
            {
            case CORE:
                return true;

            case BORDER:
            case GUARD:
                return a.direction & b.direction;
            }
        }
        else
            return false;
    }

    bool is_superset_of( Access const & other ) const
    {
        if(
            this->area == other.area &&
            this->mode.is_superset_of( other.mode )
        )
        {
            switch( this->area )
            {
            case CORE:
                return true;

            case BORDER:
            case GUARD:
                return (this->direction & other.direction) && other.direction >= this->direction;
            }
        }
        else
            return false;
    }

    friend bool operator==(Access const& a, Access const & b)
    {
        return a.area == b.area && a.direction == b.direction;
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
    //protected:
    uint32_t area;
    std::vector< uint32_t > directions;

    friend Buffer;
    friend class rg::trait::BuildProperties< ReadGuard >;

public:
    ReadGuard( std::shared_ptr< Buffer > const & obj )
        : ReadGuardBase< Buffer >( obj )
        , area{ CORE + BORDER + GUARD }
        , directions{
              TOP+LEFT, TOP, TOP+RIGHT,
              RIGHT, BOTTOM+RIGHT,
              BOTTOM, BOTTOM+LEFT, LEFT
          }
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
    std::vector< uint32_t > directions;

    friend Buffer;
    friend class rg::trait::BuildProperties< WriteGuard >;

    WriteGuard( std::shared_ptr< Buffer > const & obj )
        : WriteGuardBase< Buffer >( obj )
        , area{ CORE + BORDER + GUARD }
        , directions{
              TOP+LEFT, TOP, TOP+RIGHT,
              RIGHT, BOTTOM+RIGHT,
              BOTTOM, BOTTOM+LEFT, LEFT
          }
    {}

    ReadGuard< Buffer > read() const { return *this; }
    WriteGuard write() const { return *this; }
};

} // namespace data
} // namespace buffer

/**
 * GridBuffer represents a DIM-dimensional buffer which exists on the host as well as on the device.
 *
 * GridBuffer combines a HostBuffer and a DeviceBuffer with equal sizes.
 * Additionally, it allows sending data from and receiving data to these buffers.
 * Buffers consist of core data which may be surrounded by border data.
 *
 * @tparam TYPE datatype for internal Host- and DeviceBuffer
 * @tparam DIM dimension of the buffers
 * @tparam BORDERTYPE optional type for border data in the buffers. TYPE is used by default.
 */
template <class TYPE, std::size_t T_dim, typename BORDERTYPE>
class GridBuffer
    : public HostDeviceBuffer<TYPE, T_dim, grid_buffer::data::Access>
{
    typedef HostDeviceBuffer<TYPE, T_dim, grid_buffer::data::Access> Parent;

public:

    typedef typename Parent::DataBoxType DataBoxType;

    /**
     * Constructor.
     *
     * @param gridLayout layout of the buffers, including border-cells
     * @param sizeOnDevice if true, size information exists on device, too.
     */
    GridBuffer(const GridLayout<T_dim>& gridLayout, bool sizeOnDevice = false) :
    Parent(gridLayout.getDataSpace(), sizeOnDevice),
    gridLayout(gridLayout),
    hasOneExchange(false),
    maxExchange(0)
    {
        init();
    }

    /**
     * Constructor.
     *
     * @param dataSpace DataSpace representing buffer size without border-cells
     * @param sizeOnDevice if true, internal buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     */
    GridBuffer(DataSpace<T_dim> const & dataSpace, bool sizeOnDevice = false) :
        Parent(dataSpace, sizeOnDevice),
        gridLayout(dataSpace),
        hasOneExchange(false),
        maxExchange(0)
    {
        init();
    }

    /**
     * Constructor.
     *
     * @param otherDeviceBuffer DeviceBuffer which should be used instead of creating own DeviceBuffer
     * @param gridLayout layout of the buffers, including border-cells
     * @param sizeOnDevice if true, internal buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     */
    GridBuffer(DeviceBuffer<TYPE, T_dim, grid_buffer::data::Access> & otherDeviceBuffer,
               GridLayout<T_dim> const & gridLayout,
               bool sizeOnDevice = false) :
        Parent(otherDeviceBuffer, gridLayout.getDataSpace(), sizeOnDevice),
        gridLayout(gridLayout),
        hasOneExchange(false),
        maxExchange(0)
    {
        init();
    }

    GridBuffer(
               HostBuffer<TYPE, T_dim, grid_buffer::data::Access>& otherHostBuffer,
               const DataSpace<T_dim>& offsetHost,
               DeviceBuffer<TYPE, T_dim, grid_buffer::data::Access>& otherDeviceBuffer,
               const DataSpace<T_dim>& offsetDevice,
               const GridLayout<T_dim>& gridLayout,
               bool sizeOnDevice = false) :
    Parent(otherHostBuffer, offsetHost, otherDeviceBuffer, offsetDevice, gridLayout.getDataSpace(), sizeOnDevice),
    gridLayout(gridLayout),
    hasOneExchange(false),
    maxExchange(0)
    {
        init();
    }

    /**
     * Destructor.
     */
    virtual ~GridBuffer()
    {
        for (uint32_t i = 0; i < 27; ++i)
        {
            __delete(sendExchanges[i]);
            __delete(receiveExchanges[i]);
        };
    }

    /**
     * Add Exchange in GridBuffer memory space.
     *
     * An Exchange is added to this GridBuffer. The exchange buffers use
     * the same memory as this GridBuffer.
     *
     * @param dataPlace place where received data is stored [GUARD | BORDER]
     *        if dataPlace=GUARD than copy other BORDER to my GUARD
     *        if dataPlace=BORDER than copy other GUARD to my BORDER
     * @param receive a Mask which describes the directions for the exchange
     * @param guardingCells number of guarding cells in each dimension
     * @param communicationTag unique tag/id for communication
     * @param sizeOnDeviceSend if true, internal send buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     * @param sizeOnDeviceReceive if true, internal receive buffers must store their
     *        size additionally on the device
     */
    void addExchange(uint32_t dataPlace, const Mask &receive, DataSpace<T_dim> guardingCells, uint32_t communicationTag, bool sizeOnDeviceSend, bool sizeOnDeviceReceive )
    {

        if (hasOneExchange && (communicationTag != lastUsedCommunicationTag))
            throw std::runtime_error("It is not allowed to give the same GridBuffer different communicationTags");

        lastUsedCommunicationTag = communicationTag;

        receiveMask = receiveMask + receive;
        sendMask = this->receiveMask.getMirroredMask();
        Mask send = receive.getMirroredMask();



        for (uint32_t ex = 1; ex< -12 * (int) T_dim + 6 * (int) T_dim * (int) T_dim + 9; ++ex)
        {
            if (send.isSet(ex))
            {
                uint32_t uniqCommunicationTag = (communicationTag << 5) | ex;

                if (!hasOneExchange && !privateGridBuffer::UniquTag::getInstance().isTagUniqu(uniqCommunicationTag))
                {
                    std::stringstream message;
                    message << "unique exchange communication tag ("
                        << uniqCommunicationTag << ") witch is created from communicationTag ("
                        << communicationTag << ") already used for other GridBuffer exchange";
                    throw std::runtime_error(message.str());
                }
                hasOneExchange = true;

                if (sendExchanges[ex] != nullptr)
                {
                    throw std::runtime_error("Exchange already added!");
                }

                maxExchange = std::max(maxExchange, ex + 1u);
                sendExchanges[ex] = new ExchangeIntern< BORDERTYPE, T_dim, grid_buffer::data::Access > (this->getDeviceBuffer(), gridLayout, guardingCells,
                                                                          (ExchangeType) ex, uniqCommunicationTag,
                                                                          dataPlace == GUARD ? BORDER : GUARD, sizeOnDeviceSend);
                ExchangeType recvex = Mask::getMirroredExchangeType(ex);
                maxExchange = std::max(maxExchange, recvex + 1u);
                receiveExchanges[recvex] =
                    new ExchangeIntern< BORDERTYPE, T_dim, grid_buffer::data::Access > (
                                                          this->getDeviceBuffer(),
                                                          gridLayout,
                                                          guardingCells,
                                                          recvex,
                                                          uniqCommunicationTag,
                                                          dataPlace == GUARD ? GUARD : BORDER,
                                                          sizeOnDeviceReceive);
            }
        }
    }

    /**
     * Add Exchange in GridBuffer memory space.
     *
     * An Exchange is added to this GridBuffer. The exchange buffers use
     * the same memory as this GridBuffer.
     *
     * @param dataPlace place where received data is stored [GUARD | BORDER]
     *        if dataPlace=GUARD than copy other BORDER to my GUARD
     *        if dataPlace=BORDER than copy other GUARD to my BORDER
     * @param receive a Mask which describes the directions for the exchange
     * @param guardingCells number of guarding cells in each dimension
     * @param communicationTag unique tag/id for communication
     * @param sizeOnDevice if true, internal buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     */
    void addExchange(uint32_t dataPlace, const Mask &receive, DataSpace<T_dim> guardingCells, uint32_t communicationTag, bool sizeOnDevice = false)
    {
        addExchange( dataPlace, receive, guardingCells, communicationTag, sizeOnDevice, sizeOnDevice );
    }

    /**
     * Add Exchange in dedicated memory space.
     *
     * An Exchange is added to this GridBuffer. The exchange buffers use
     * the their own memory instead of using the GridBuffer's memory space.
     *
     * @param receive a Mask which describes the directions for the exchange
     * @param dataSpace size of the newly created exchange buffer in each dimension
     * @param communicationTag unique tag/id for communication
     * @param sizeOnDeviceSend if true, internal send buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     * @param sizeOnDeviceReceive if true, internal receive buffers must store their
     *        size additionally on the device
     */
    void addExchangeBuffer(const Mask &receive, const DataSpace<T_dim> &dataSpace, uint32_t communicationTag, bool sizeOnDeviceSend, bool sizeOnDeviceReceive )
    {

        if (hasOneExchange && (communicationTag != lastUsedCommunicationTag))
            throw std::runtime_error("It is not allowed to give the same GridBuffer different communicationTags");
        lastUsedCommunicationTag = communicationTag;


        /*don't create buffer with 0 (zero) elements*/
        if (dataSpace.productOfComponents() != 0)
        {
            receiveMask = receiveMask + receive;
            sendMask = this->receiveMask.getMirroredMask();
            Mask send = receive.getMirroredMask();
            for (uint32_t ex = 1; ex < 27; ++ex)
            {
                if (send.isSet(ex))
                {
                    uint32_t uniqCommunicationTag = (communicationTag << 5) | ex;
                    if (!hasOneExchange && !privateGridBuffer::UniquTag::getInstance().isTagUniqu(uniqCommunicationTag))
                    {
                        std::stringstream message;
                        message << "unique exchange communication tag ("
                            << uniqCommunicationTag << ") witch is created from communicationTag ("
                            << communicationTag << ") already used for other GridBuffer exchange";
                        throw std::runtime_error(message.str());
                    }
                    hasOneExchange = true;

                    if (sendExchanges[ex] != nullptr)
                    {
                        throw std::runtime_error("Exchange already added!");
                    }

                    //GridLayout<DIM> memoryLayout(size);
                    maxExchange = std::max(maxExchange, ex + 1u);
                    sendExchanges[ex] = new ExchangeIntern< BORDERTYPE, T_dim, grid_buffer::data::Access > (/*memoryLayout*/ dataSpace,
                                                                              ex, uniqCommunicationTag, sizeOnDeviceSend);

                    ExchangeType recvex = Mask::getMirroredExchangeType(ex);
                    maxExchange = std::max(maxExchange, recvex + 1u);
                    receiveExchanges[recvex] = new ExchangeIntern< BORDERTYPE, T_dim, grid_buffer::data::Access > (/*memoryLayout*/ dataSpace,
                                                                                     recvex, uniqCommunicationTag, sizeOnDeviceReceive);
                }
            }
        }
    }

    /**
     * Add Exchange in dedicated memory space.
     *
     * An Exchange is added to this GridBuffer. The exchange buffers use
     * the their own memory instead of using the GridBuffer's memory space.
     *
     * @param receive a Mask which describes the directions for the exchange
     * @param dataSpace size of the newly created exchange buffer in each dimension
     * @param communicationTag unique tag/id for communication
     * @param sizeOnDevice if true, internal buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     */
    void addExchangeBuffer(const Mask &receive, const DataSpace<T_dim> &dataSpace, uint32_t communicationTag, bool sizeOnDevice = false )
    {
        addExchangeBuffer( receive, dataSpace, communicationTag, sizeOnDevice, sizeOnDevice );
    }

    /**
     * Returns whether this GridBuffer has an Exchange for sending in ex direction.
     *
     * @param ex exchange direction to query
     * @return true if send exchanges with ex direction exist, otherwise false
     */
    bool hasSendExchange(uint32_t ex) const
    {
        return ( (sendExchanges[ex] != nullptr) && (getSendMask().isSet(ex)));
    }

    /**
     * Returns whether this GridBuffer has an Exchange for receiving from ex direction.
     *
     * @param ex exchange direction to query
     * @return true if receive exchanges with ex direction exist, otherwise false
     */
    bool hasReceiveExchange(uint32_t ex) const
    {
        return ( (receiveExchanges[ex] != nullptr) && (getReceiveMask().isSet(ex)));
    }

    /**
     * Returns the Exchange for sending data in ex direction.
     *
     * Returns an Exchange which for sending data from
     * this GridBuffer in the direction described by ex.
     *
     * @param ex the direction to query
     * @return the Exchange for sending data
     */
    Exchange<BORDERTYPE, T_dim, grid_buffer::data::Access>& getSendExchange(uint32_t ex) const
    {
        return *sendExchanges[ex];
    }

    /**
     * Returns the Exchange for receiving data from ex direction.
     *
     * Returns an Exchange which for receiving data to
     * this GridBuffer from the direction described by ex.
     *
     * @param ex the direction to query
     * @return the Exchange for receiving data
     */
    Exchange<BORDERTYPE, T_dim, grid_buffer::data::Access>& getReceiveExchange(uint32_t ex) const
    {
        return *receiveExchanges[ex];
    }

    /**
     * Returns the Mask describing send exchanges
     *
     * @return Mask for send exchanges
     */
    Mask getSendMask() const
    {
        return (Environment<T_dim>::get().EnvironmentController().getCommunicationMask() & sendMask);
    }

    /**
     * Returns the Mask describing receive exchanges
     *
     * @return Mask for receive exchanges
     */
    Mask getReceiveMask() const
    {
        return (Environment<T_dim>::get().EnvironmentController().getCommunicationMask() & receiveMask);
    }

    /**
     * Starts sync data from own device buffer to neighbor device buffer.
     *
     * Asynchronously starts synchronization data from internal DeviceBuffer using added
     * Exchange buffers.
     *
     */
    void communication()
    {
        for (uint32_t i = 0; i < maxExchange; ++i)
            send( i );
        for (uint32_t i = 0; i < maxExchange; ++i)
            receive( i );
    }

    void send(uint32_t sendEx)
    {
        if (hasSendExchange(sendEx))
            sendExchanges[sendEx]->startSend();
    }

    void receive(uint32_t recvEx)
    {
        if (hasReceiveExchange(recvEx))
            receiveExchanges[recvEx]->startReceive();
    }

    /**
     * Returns the GridLayout describing this GridBuffer.
     *
     * @return the layout of this buffer
     */
    GridLayout<T_dim> getGridLayout()
    {
        return gridLayout;
    }

private:

    friend class Environment<T_dim>;

    void init()
    {
        for (uint32_t i = 0; i < 27; ++i)
        {
            sendExchanges[i] = nullptr;
            receiveExchanges[i] = nullptr;
        }
    }

protected:
    /*if we have one exchange we don't check if communicationTag has been used before*/
    bool hasOneExchange;
    uint32_t lastUsedCommunicationTag;
    GridLayout<T_dim> gridLayout;

    Mask sendMask;
    Mask receiveMask;

    ExchangeIntern<BORDERTYPE, T_dim, grid_buffer::data::Access>* sendExchanges[27];
    ExchangeIntern<BORDERTYPE, T_dim, grid_buffer::data::Access>* receiveExchanges[27];

    uint32_t maxExchange; //use max exchanges and run over the array is faster as use set from stl
};

template < typename T_Item, std::size_t T_dim, typename T_Bordertype >
struct BufferResource< GridBuffer< T_Item, T_dim, T_Bordertype > >
    : BufferResource< HostDeviceBuffer< T_Item, T_dim, grid_buffer::data::Access > >
{
    template < typename... Args >
    BufferResource( Args&&... args )
        : BufferResource< HostDeviceBuffer< T_Item, T_dim, grid_buffer::data::Access > >(std::static_pointer_cast< HostDeviceBuffer< T_Item, T_dim, grid_buffer::data::Access > >( std::make_shared< GridBuffer< T_Item, T_dim, T_Bordertype > >( std::forward<Args>(args)... )))
    {}    
};

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
        for(auto dir : buf.directions)
            builder.add(
                buf.get()->access_data(
                    pmacc::mem::grid_buffer::data::Access {
                        rg::access::IOAccess::read,
                        buf.area,
                        dir
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
        pmacc::mem::buffer::data::ReadGuard< Buffer > const & buf
    )
    {
        for(auto dir : buf.directions)
            builder.add(
                buf.get()->access_data(
                    pmacc::mem::grid_buffer::data::Access {
                        rg::access::IOAccess::write,
                        buf.area,
                        dir
                    }));
       
    }
};

} // namespace trait

} // namespace redGrapes



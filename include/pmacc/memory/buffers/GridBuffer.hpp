/* Copyright 2013-2020 Rene Widera, Benjamin Worpitz, Alexander Grund,
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


#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>

#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/buffers/gridBuffer/AccessPolicy.hpp>
#include <pmacc/memory/buffers/gridBuffer/Exchange.hpp>

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

} // namespace privateGridBuffer


/*!
 * GridBuffer represents a `T_dim`-dimensional buffer which exists on the host as well as on the device.
 *
 * GridBuffer combines a HostBuffer and a DeviceBuffer with equal sizes.
 * Additionally, it allows sending data from and receiving data to these buffers.
 * Buffers consist of core data which may be surrounded by border data.
 *
 * @tparam T_Item datatype for internal Host- and DeviceBuffer
 * @tparam T_dim dimension of the buffers
 * @tparam T_BorderItem optional type for border data in the buffers. T_Item is used by default.
 */
template<
    typename T_Item,
    std::size_t T_dim,
    typename T_BorderItem = T_Item
>
struct GridBuffer
    : HostDeviceBuffer<
        T_Item,
        T_dim,
        grid_buffer::data::Access
    >
{
    using Exchange = ExchangeBuffer< T_BorderItem, T_dim, grid_buffer::data::Access >;

    /*!
     * Constructor.
     *
     * @param gridLayout layout of the buffers, including border-cells
     * @param sizeOnDevice if true, size information exists on device, too.
     */
    GridBuffer(
        GridLayout<T_dim> const & gridLayout,
        bool sizeOnDevice = false
    ) :
        HostDeviceBuffer<
            T_Item, T_dim, grid_buffer::data::Access
        >(
            gridLayout.getDataSpace(),
            sizeOnDevice
        ),
        gridLayout( gridLayout ),
        useMpiDirect( false )
    {}

    /*!
     * Constructor.
     *
     * @param dataSpace DataSpace representing buffer size without border-cells
     * @param sizeOnDevice if true, internal buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     */
    GridBuffer(
        DataSpace< T_dim > const & dataSpace,
        bool sizeOnDevice = false
    ) :
        HostDeviceBuffer<
            T_Item, T_dim, grid_buffer::data::Access
        >(
            dataSpace,
            sizeOnDevice
        ),
        gridLayout( dataSpace )
    {}

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
    void addExchange(
        uint32_t dataPlace,
        Mask const & receive,
        DataSpace< T_dim > const & guardingCells,
        uint32_t communicationTag,
        bool sizeOnDeviceSend = false,
        bool sizeOnDeviceRecv = false
    )
    {        
        if (hasOneExchange && (communicationTag != lastUsedCommunicationTag))
            throw std::runtime_error("It is not allowed to give the same GridBuffer different communicationTags");

        lastUsedCommunicationTag = communicationTag;

        receiveMask = receiveMask + receive;
        sendMask = this->receiveMask.getMirroredMask();
        Mask send = receive.getMirroredMask();

        auto n_ex = -12 * (int) T_dim + 6 * (int) T_dim * (int) T_dim + 9;
        for (uint32_t ex = 1; ex < n_ex; ++ex)
        {
            if (send.isSet(ex))
            {
                uint32_t uniqCommunicationTag = (communicationTag << 5) | ex;

                if (!hasOneExchange && !privateGridBuffer::UniquTag::getInstance().isTagUniqu(uniqCommunicationTag))
                {
                    std::stringstream message;
                    message << "unique exchange communication tag ("
                        << uniqCommunicationTag << ") which is created from communicationTag ("
                        << communicationTag << ") already used for other GridBuffer exchange";
                    throw std::runtime_error(message.str());
                }

                hasOneExchange = true;

                auto sendex = ex;
                if ( ! sendExchanges[sendex] )
                {
                    sendExchanges[sendex].emplace(
                    get_device_exchange(
                        sendex,
                        dataPlace == GUARD ? BORDER : GUARD,
                        guardingCells
                    ),
                    sendex,
                    uniqCommunicationTag,
                    useMpiDirect,
                    sizeOnDeviceSend);
                }
                else
                    throw std::runtime_error("Exchange already added!");

                auto recvex = Mask::getMirroredExchangeType(ex);
                if ( ! recvExchanges[recvex] )
                {
                    recvExchanges[recvex].emplace(
                        get_device_exchange(
                            recvex,
                            dataPlace == GUARD ? GUARD : BORDER,
                            guardingCells
                        ),
                        recvex,
                        uniqCommunicationTag,
                        useMpiDirect,
                        sizeOnDeviceRecv);
                }
                else
                    throw std::runtime_error("Exchange already added!");
                
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
     * @param sizeOnDeviceSend if true, internal send buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     * @param sizeOnDeviceReceive if true, internal receive buffers must store their
     *        size additionally on the device
     */
    void addExchangeBuffer(
        Mask const & receive,
        DataSpace< T_dim > const & dataSpace,
        uint32_t communicationTag,
        bool sizeOnDeviceSend,
        bool sizeOnDeviceReceive
    )
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
                            << uniqCommunicationTag << ") which is created from communicationTag ("
                            << communicationTag << ") already used for other GridBuffer exchange";
                        throw std::runtime_error(message.str());
                    }
                    hasOneExchange = true;

                    if (sendExchanges[ex] != nullptr)
                    {
                        throw std::runtime_error("Exchange already added!");
                    }

                    //GridLayout<DIM> memoryLayout(size);
                    //maxExchange = std::max(maxExchange, ex + 1u);


                    auto sendex = ex;
                    sendExchanges[sendex].emplace(
                        DeviceBuffer< T_BorderItem, T_dim, grid_buffer::data::Access >( dataSpace, sizeOnDeviceSend ),
                        sendex,
                        uniqCommunicationTag,
                        useMpiDirect,
                        sizeOnDeviceSend
                    );

                    ExchangeType recvex = Mask::getMirroredExchangeType(ex);
                    //maxExchange = std::max(maxExchange, recvex + 1u);

                    recvExchanges[recvex].emplace(
                        DeviceBuffer< T_BorderItem, T_dim, grid_buffer::data::Access >( dataSpace, sizeOnDeviceReceive ),
                        recvex,
                        uniqCommunicationTag,
                        useMpiDirect,
                        sizeOnDeviceReceive
                    );
                }
            }
        }
    }

    device_buffer::WriteGuard<
        T_Item,
        T_dim,
        grid_buffer::data::Access
    >
    get_device_exchange(
        uint32_t exchangeType,
        uint32_t dataPlace,
        DataSpace<T_dim> guardingCells
    )
    {
        return this->device().sub_area(
            exchange::exchangeTypeToOffset< T_dim >(
                exchangeType,
                gridLayout,
                guardingCells,
                dataPlace == GUARD ? GUARD : BORDER
            ),
            exchange::exchangeTypeToDataSpace< T_dim >(
                exchangeType,
                gridLayout,
                guardingCells
            )
        );
    }

    /*!
     * Starts sync data from own device buffer to neighbor device buffer.
     *
     * Asynchronously starts synchronization data from internal DeviceBuffer using added
     * Exchange buffers.
     */
    void communication()
    {
        for( auto & exchangeBuffer : sendExchanges )
            if( exchangeBuffer )
                exchangeBuffer->send();

        for( auto & exchangeBuffer : recvExchanges )
            if( exchangeBuffer )
                exchangeBuffer->recv();
    }

    /*!
     * Returns whether this GridBuffer has an Exchange for sending in ex direction.
     *
     * @param ex exchange direction to query
     * @return true if send exchanges with ex direction exist, otherwise false
     */
    bool hasSendExchange(uint32_t ex) const
    {
        return ( sendExchanges[ex] && (getSendMask().isSet(ex)));
    }

    /*!
     * Returns whether this GridBuffer has an Exchange for receiving from ex direction.
     *
     * @param ex exchange direction to query
     * @return true if receive exchanges with ex direction exist, otherwise false
     */
    bool hasReceiveExchange(uint32_t ex) const
    {
        return ( recvExchanges[ex] && (getReceiveMask().isSet(ex)));
    }

    /*!
     * Returns the Exchange for sending data in ex direction.
     *
     * Returns an Exchange which for sending data from
     * this GridBuffer in the direction described by ex.
     *
     * @param ex the direction to query
     * @return the Exchange for sending data
     */
    std::optional< Exchange >
    getSendExchange(uint32_t ex) const
    {
        return sendExchanges[ex];
    }

    /*!
     * Returns the Exchange for receiving data from ex direction.
     *
     * Returns an Exchange which for receiving data to
     * this GridBuffer from the direction described by ex.
     *
     * @param ex the direction to query
     * @return the Exchange for receiving data
     */
    std::optional< Exchange >
    getReceiveExchange(uint32_t ex) const
    {
        return recvExchanges[ex];
    }

    /*!
     * Returns the Mask describing send exchanges
     *
     * @return Mask for send exchanges
     */
    Mask getSendMask() const
    {
        return (Environment<T_dim>::get().EnvironmentController().getCommunicationMask() & sendMask);
    }

    /*!
     * Returns the Mask describing receive exchanges
     *
     * @return Mask for receive exchanges
     */
    Mask getReceiveMask() const
    {
        return (Environment<T_dim>::get().EnvironmentController().getCommunicationMask() & receiveMask);
    }

    GridLayout< T_dim > getGridLayout()
    {
        return gridLayout;
    }
    
private:
    GridLayout< T_dim > gridLayout;

    bool useMpiDirect;

    /*if we have one exchange we don't check if communicationTag has been used before*/
    bool hasOneExchange = false;
    uint32_t lastUsedCommunicationTag;

    Mask sendMask;
    Mask receiveMask;

    std::array<
        std::optional< Exchange >,
        27
    > sendExchanges, recvExchanges;
};

} // namespace mem

} // namespace pmacc


/* Copyright 2013-2020 Rene Widera, Benjamin Worpitz, Michael Sippel
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

#include <pmacc/assert.hpp>
#include <pmacc/types.hpp>

#include <memory>

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>

#include <pmacc/memory/buffers/HostBuffer.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>
#include <pmacc/memory/buffers/copy/DeviceToDevice.hpp>
#include <pmacc/memory/buffers/copy/DeviceToHost.hpp>
#include <pmacc/memory/buffers/copy/HostToDevice.hpp>

namespace pmacc
{
namespace mem
{

namespace exchange
{

/**
 * specifies in returned DataSpace which dimensions exchange data
 * @param exchange the exchange mask
 * @return DIM1 DataSpace of size 3 where 1 means exchange, 0 means no exchange
 */
template < std::size_t T_dim >
DataSpace< T_dim > exchangeTypeToDim( uint32_t exchange ) noexcept
{
    DataSpace< T_dim > result;

    Mask exchangeMask(exchange);

    if (exchangeMask.containsExchangeType(LEFT) || exchangeMask.containsExchangeType(RIGHT))
        result[0] = 1;

    if (T_dim > DIM1 && (exchangeMask.containsExchangeType(TOP) || exchangeMask.containsExchangeType(BOTTOM)))
        result[1] = 1;

    if (T_dim > DIM2 && (exchangeMask.containsExchangeType(FRONT) || exchangeMask.containsExchangeType(BACK)))
        result[2] = 1;

    return result;
}

template < std::size_t T_dim >
DataSpace< T_dim >
exchangeTypeToOffset(
    uint32_t exchange,
    GridLayout< T_dim > const & memoryLayout,
    DataSpace< T_dim > guardingCells,
    uint32_t area
)
{
    DataSpace< T_dim > size = memoryLayout.getDataSpace();
    DataSpace< T_dim > border = memoryLayout.getGuard();

    DataSpace< T_dim > tmp_offset;
    Mask mask(exchange);

    if ( T_dim >= DIM1 )
    {
        if (mask.containsExchangeType(RIGHT))
        {
            tmp_offset[0] = size[0] - border[0] - guardingCells[0];
            if( area == GUARD )
                tmp_offset[0] += guardingCells[0];

            /* std::cout<<"offset="<<tmp_offset[0]<<"border"<<border[0]<<std::endl;*/
        }
        else
        {
            tmp_offset[0] = border[0];
            if (area == GUARD && mask.containsExchangeType(LEFT))
                        tmp_offset[0] -= guardingCells[0];
        }
    }
    if( T_dim >= DIM2 )
    {
        if (mask.containsExchangeType(BOTTOM))
        {
            tmp_offset[1] = size[1] - border[1] - guardingCells[1];
            if (area == GUARD)
                tmp_offset[1] += guardingCells[1];
        }
        else
        {
            tmp_offset[1] = border[1];
            if (area == GUARD && mask.containsExchangeType(TOP))
                tmp_offset[1] -= guardingCells[1];
        }
    }
    if( T_dim == DIM3 )
    {
        if (mask.containsExchangeType(BACK))
        {
            tmp_offset[2] = size[2] - border[2] - guardingCells[2];
            if (area == GUARD)
                tmp_offset[2] += guardingCells[2];
        }
        else /*all other begin from front*/
        {
            tmp_offset[2] = border[2];
            if (area == GUARD && mask.containsExchangeType(FRONT))
                tmp_offset[2] -= guardingCells[2];
        }
    }

    return tmp_offset;
}

template < std::size_t T_dim >
DataSpace< T_dim >
exchangeTypeToDataSpace(
    uint32_t exchangeType,
    GridLayout< T_dim > const & gridLayout,
    DataSpace< T_dim > guardingCells
)
{
    PMACC_ASSERT(! guardingCells.isOneDimensionGreaterThan(gridLayout.getGuard()) );

    DataSpace< T_dim > tmp_size = gridLayout.getDataSpaceWithoutGuarding();
    DataSpace< T_dim > exchangeDimensions = exchangeTypeToDim< T_dim >( exchangeType );

    for( uint32_t dim = 0; dim < T_dim; dim++ )
    {
        if( T_dim > dim && exchangeDimensions[dim] == 1 )
            tmp_size[dim] = guardingCells[dim];
    }

    return tmp_size;
}

} // namespace exchange

struct Exchange
{
    uint32_t exchangeType;
    uint32_t communicationTag;

    /**
     * Returns the value used for tagging ('naming') communicated messages
     *
     * @return the communication tag
     */
    uint32_t getCommunicationTag() const
    {
        return communicationTag;
    }

    template< typename BufferResource >
    void recvBuf( buffer::WriteGuard< BufferResource > const &  messageBuffer )
    {
        PMACC_ASSERT( messageBuffer.is1D() );

        // need to create a task because the Communicator
        // doesn't know about the resource
        Environment<>::task(
            [=]( auto messageBuffer )
            {
                size_t new_size = Environment<DIM2>::get()
                    .GridController()
                    .getCommunicator()
                    .recv(
                        exchangeType,
                        ( char * ) messageBuffer.data().getPointer(),
                        messageBuffer.getDataSpace().productOfComponents() * sizeof(typename BufferResource::Item),
                        communicationTag
                    );

                messageBuffer.size().set( new_size / sizeof(typename BufferResource::Item) );
            },
            TaskProperties::Builder().label("Exchange::recv()"),
            messageBuffer.write()
        );
    }

    template< typename BufferResource >
    void sendBuf( buffer::ReadGuard< BufferResource > messageBuffer )
    {
        PMACC_ASSERT( messageBuffer.is1D() );

        // need to create a task because the Communicator
        // doesn't know about the resource
        Environment<>::task(
            [=]( auto messageBuffer )
            {
                Environment<DIM2>::get()
                    .GridController()
                    .getCommunicator()
                    .send(
                        exchangeType,
                        ( char const* ) messageBuffer.data().getPointer(),
                        messageBuffer.size().get() * sizeof(typename BufferResource::Item),
                        communicationTag
                    );
            },
            TaskProperties::Builder().label("Exchange::send()"),
            messageBuffer.read()
        );
    }
};

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy
>
struct ExchangeBuffer
    : Exchange
{
    ExchangeBuffer(
        device_buffer::WriteGuard<
            T_Item, T_dim, T_DataAccessPolicy
        > const & deviceBuffer,
        uint32_t exchangeType,
        uint32_t communicationTag,
        bool useMpiDirect,
        bool sizeOnDevice
    ) :
        Exchange{ exchangeType, communicationTag },
        deviceBuffer( deviceBuffer )
    {
        if( T_dim > DIM1 )
            deviceDoubleBuffer.emplace( deviceBuffer.getDataSpace(), sizeOnDevice, true );

        if( ! useMpiDirect )
            hostBuffer.emplace( deviceBuffer.getDataSpace() );
    }

    void send()
    {
        if( deviceDoubleBuffer )
        {
            buffer::copy( deviceDoubleBuffer->write(), deviceBuffer.read() );

            if( hostBuffer )
            {
                // send over host memory
                buffer::copy( hostBuffer->write(), deviceDoubleBuffer->read() );
                this->sendBuf( *hostBuffer );
            }
            else
                // use mpi direct
                this->sendBuf( *deviceDoubleBuffer );
        }
        else
        {
            if( hostBuffer )
            {
                // send over host memory
                buffer::copy( hostBuffer->write(), deviceBuffer.read() );
                this->sendBuf( *hostBuffer );
            }
            else
                // use mpi direct
                this->sendBuf( deviceBuffer );
        }
    }

    void recv()
    {
        if( deviceDoubleBuffer )
        {
            if( hostBuffer )
            {
                this->recvBuf( *hostBuffer );
                buffer::copy( deviceDoubleBuffer->write(), hostBuffer->read() );
            }
            else
                this->recvBuf( *deviceDoubleBuffer );

            buffer::copy( deviceBuffer.write(), deviceDoubleBuffer->read() );
        }
        else
        {
            if( hostBuffer )
            {
                this->recvBuf( *hostBuffer );
                buffer::copy( deviceBuffer.write(), hostBuffer->read() );
            }
            else
                this->recvBuf( deviceBuffer );
        }
    }

    auto getHostBuffer()
    {
        return hostBuffer;
    }

    auto getDeviceBuffer()
    {
        return deviceBuffer;
    }
    
private:

    //! access to device data (from GridBuffer)
    device_buffer::WriteGuard<
        T_Item,
        T_dim,
        T_DataAccessPolicy
    > deviceBuffer;

    //! when no MPI-direct is used
    std::optional<
        HostBuffer<
            T_Item,
            T_dim
        >
    > hostBuffer;

    //! serialization buffer
    std::optional<
        DeviceBuffer<
            T_Item,
            T_dim
        >
    > deviceDoubleBuffer;
};

} // namespace mem

} // namespace pmacc


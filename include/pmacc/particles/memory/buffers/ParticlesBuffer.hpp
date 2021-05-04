/* Copyright 2013-2020 Axel Huebl, Felix Schmitt, Rene Widera,
 *                     Benjamin Worpitz, Michael Sippel
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

#include "pmacc/particles/frame_types.hpp"
#include "pmacc/memory/buffers/GridBuffer.hpp"
#include "pmacc/particles/memory/boxes/ParticlesBox.hpp"
#include "pmacc/dimensions/GridLayout.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/particles/memory/buffers/StackExchangeBuffer.hpp"
#include "pmacc/particles/memory/dataTypes/SuperCell.hpp"

#include "pmacc/math/Vector.hpp"

#include "pmacc/particles/boostExtension/InheritGenerators.hpp"
#include "pmacc/meta/conversion/MakeSeq.hpp"


#include <boost/mpl/vector.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/back_inserter.hpp>

#include "pmacc/particles/memory/frames/Frame.hpp"
#include "pmacc/particles/Identifier.hpp"
#include "pmacc/particles/memory/dataTypes/StaticArray.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include "pmacc/particles/ParticleDescription.hpp"
#include "pmacc/particles/memory/dataTypes/ListPointer.hpp"

#include <memory>

namespace pmacc
{

namespace particles_buffer
{
template<typename T_ParticleDescription, class SuperCellSize_, typename T_DeviceHeap, unsigned DIM>
struct HostGuard;

template<typename T_ParticleDescription, class SuperCellSize_, typename T_DeviceHeap, unsigned DIM>
struct DeviceGuard;
}

/**
 * Describes DIM-dimensional buffer for particles data on the host.
 *
 * @tParam T_ParticleDescription Object which describe a frame @see ParticleDescription.hpp
 * @tparam SuperCellSize_ TVec which descripe size of a superce
 * @tparam DIM dimension of the buffer (1-3)
 */
template<typename T_ParticleDescription, class SuperCellSize_, typename T_DeviceHeap, unsigned DIM>
class ParticlesBuffer
{
public:

    /** create static array
     */
    template< uint32_t T_size >
    struct OperatorCreatePairStaticArray
    {

        template<typename X>
        struct apply
        {
            typedef bmpl::pair<
                X,
                StaticArray<
                    typename traits::Resolve<X>::type::type,
                    bmpl::integral_c<uint32_t, T_size>
                >
            > type;
        };
    };

    /** type of the border frame management object
     *
     * contains:
     *   - superCell position of the border frames inside a given range
     *   - start position inside the exchange stack for frames
     *   - number of frames corresponding to the superCell position
     */
    typedef ExchangeMemoryIndex<
        vint_t,
        DIM - 1
    > BorderFrameIndex;

    typedef SuperCellSize_ SuperCellSize;

    typedef typename MakeSeq<
        typename T_ParticleDescription::ValueTypeSeq,
        localCellIdx,
        multiMask
    >::type ParticleAttributeList;

    typedef typename MakeSeq<
        typename T_ParticleDescription::ValueTypeSeq,
        localCellIdx
    >::type ParticleAttributeListBorder;

    typedef
    typename ReplaceValueTypeSeq<
        T_ParticleDescription,
        ParticleAttributeList
    >::type FrameDescriptionWithManagementAttributes;

    /** double linked list pointer */
    typedef
    typename MakeSeq<
        PreviousFramePtr<>,
        NextFramePtr<>
    >::type LinkedListPointer;

    /* extent particle description with pointer to a frame*/
    typedef typename ReplaceFrameExtensionSeq<
        FrameDescriptionWithManagementAttributes,
        LinkedListPointer
    >::type FrameDescription;

    /** frame definition
     *
     * a group of particles is stored as frame
     */
    typedef Frame<
        OperatorCreatePairStaticArray<
            pmacc::math::CT::volume< SuperCellSize >::type::value
        >,
        FrameDescription
    > FrameType;

    typedef typename ReplaceValueTypeSeq<
        T_ParticleDescription,
        ParticleAttributeListBorder
    >::type FrameDescriptionBorder;

    /** frame which is used to communicate particles to neighbors
     *
     * - each frame contains only one particle
     * - local administration attributes of a particle are removed
     */
    typedef Frame<
        OperatorCreatePairStaticArray< 1u >,
        FrameDescriptionBorder
    > FrameTypeBorder;

    typedef SuperCell<FrameType> SuperCellType;

    typedef T_DeviceHeap DeviceHeap;

    /* Type of the particle box which particle buffer create */
    typedef ParticlesBox< FrameType, typename DeviceHeap::AllocatorHandle, DIM > ParticlesBoxType;

private:

    /* this enum is used only for internal calculations */
    enum
    {
        SizeOfOneBorderElement = (sizeof (FrameTypeBorder) + sizeof (BorderFrameIndex))
    };

public:

    /**
     * Constructor.
     *
     * @param deviceHeap device heap memory allocator
     * @param layout number of cell per dimension
     * @param superCellSize size of one super cell
     * @param gpuMemory how many memory on device is used for this instance (in byte)
     */
    ParticlesBuffer(const std::shared_ptr<DeviceHeap>& deviceHeap, DataSpace<DIM> layout, DataSpace<DIM> superCellSize) :
        m_deviceHeap(deviceHeap),
        superCellSize(superCellSize),
        gridSize(layout),
        exchangeMemoryIndexer( DataSpace< DIM1 >( 0 ) ),
        framesExchanges( DataSpace< DIM1 >( 0 ) ),
        superCells( DataSpace< DIM >( layout / superCellSize ) )
    {
        reset();
    }

    ParticlesBuffer( ParticlesBuffer const & other ) :
        m_deviceHeap(other.m_deviceHeap),
        superCellSize( other.superCellSize ),
        gridSize(other.gridSize),
        exchangeMemoryIndexer(other.exchangeMemoryIndexer),
        framesExchanges(other.framesExchanges),
        superCells(other.superCells)
    {}

    /**
     * Destructor.
     */
    virtual ~ParticlesBuffer() {}

    auto host()
    {
        return particles_buffer::HostGuard<T_ParticleDescription, SuperCellSize, DeviceHeap, DIM>( *this );
    }

    auto device()
    {
        return particles_buffer::DeviceGuard<T_ParticleDescription, SuperCellSize, DeviceHeap, DIM>( *this );
    }

    /**
     * Resets all internal buffers.
     */
    void reset()
    {
        mem::buffer::fill( superCells.device(), SuperCellType() );
        mem::buffer::fill( superCells.host(), SuperCellType() );
    }

    /**
     * Adds an exchange buffer to frames.
     *
     * @param receive Mask describing receive directions
     * @param usedMemory memory to be used for this exchange
     */
    void addExchange(Mask receive, size_t usedMemory, uint32_t communicationTag)
    {
        size_t numFrameTypeBorders = usedMemory / SizeOfOneBorderElement;

        framesExchanges.addExchangeBuffer(
            receive,
            DataSpace< DIM1 >( numFrameTypeBorders ),
            communicationTag,
            true,
            false
        );

        exchangeMemoryIndexer.addExchangeBuffer(
            receive,
            DataSpace< DIM1 >( numFrameTypeBorders ),
            communicationTag | ( 1u << (20 - 5) ),
            true,
            false
        );
    }

    /**
     * Returns if the buffer has a send exchange in ex direction.
     *
     * @param ex direction to query
     * @return true if buffer has send exchange for ex
     */
    bool hasSendExchange(uint32_t ex)
    {
        return framesExchanges.hasSendExchange(ex);
    }

    /**
     * Returns if the buffer has a receive exchange in ex direction.
     *
     * @param ex direction to query
     * @return true if buffer has receive exchange for ex
     */
    bool hasReceiveExchange(uint32_t ex)
    {
        return framesExchanges.hasReceiveExchange(ex);
    }

    StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1 > getSendExchangeStack(uint32_t ex)
    {
        return StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1 >
            (*framesExchanges.getSendExchange(ex), *exchangeMemoryIndexer.getSendExchange(ex));
    }

    StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1 > getReceiveExchangeStack(uint32_t ex)
    {
        return StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1 >
            (*framesExchanges.getReceiveExchange(ex), *exchangeMemoryIndexer.getReceiveExchange(ex));
    }

    void communication()
    {
        framesExchanges.communication();
        exchangeMemoryIndexer.communication();
    }

    void send( uint32_t direction )
    {
        framesExchanges.send( direction );
        exchangeMemoryIndexer.send( direction );
    }

    void recv( uint32_t direction )
    {
        framesExchanges.recv( direction );
        exchangeMemoryIndexer.recv( direction );
    }
    
    /**
     * Returns number of supercells in each dimension.
     *
     * @return number of supercells
     */
    DataSpace<DIM> getSuperCellsCount()
    {
        return superCells.getGridLayout().getDataSpace();
    }

    /**
     * Returns number of supercells in each dimension.
     *
     * @return number of supercells
     */
    GridLayout<DIM> getSuperCellsLayout()
    {
        return superCells.getGridLayout();
    }

    /**
     * Returns size of supercells in each dimension.
     *
     * @return size of supercells
     */
    DataSpace<DIM> getSuperCellSize()
    {
        return superCellSize;
    }

    void deviceToHost()
    {
        mem::buffer::copy( superCells.host().write(), superCells.device().read() );
    }

protected:
    mem::GridBuffer<BorderFrameIndex, DIM1> exchangeMemoryIndexer;

    mem::GridBuffer<SuperCellType, DIM> superCells;
    /*GridBuffer for hold borderFrames, we need a own buffer to create first exchanges without core memory*/
    mem::GridBuffer< FrameType, DIM1, FrameTypeBorder> framesExchanges;

    DataSpace<DIM> superCellSize;
    DataSpace<DIM> gridSize;
    std::shared_ptr<DeviceHeap> m_deviceHeap;

}; // class ParticlesBuffer


namespace particles_buffer
{

template<typename T_ParticleDescription, class T_SuperCellSize, typename T_DeviceHeap, unsigned T_dim>
struct HostGuard : private ParticlesBuffer<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim>
{
    friend class redGrapes::trait::BuildProperties< HostGuard >;

    using typename ParticlesBuffer<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim>::ParticlesBoxType;

    HostGuard(ParticlesBuffer<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim> const & b)
        : ParticlesBuffer<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim>(b) {}
    
    /**
     * Returns a ParticlesBox for host frame data.
     *
     * @return host frames ParticlesBox
     */
    ParticlesBoxType getParticlesBox( int64_t memoryOffset )
    {
        return ParticlesBoxType(
            this->superCells.host().data().getDataBox(),
            this->m_deviceHeap->getAllocatorHandle(),
            memoryOffset
        );
    }
};

template<typename T_ParticleDescription, class T_SuperCellSize, typename T_DeviceHeap, unsigned T_dim>
struct DeviceGuard : private ParticlesBuffer<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim>
{
    friend class redGrapes::trait::BuildProperties< DeviceGuard >;

    using typename ParticlesBuffer<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim>::ParticlesBoxType;

    uint32_t area;
    Mask directions;
    
    DeviceGuard(ParticlesBuffer<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim> const & b)
        : ParticlesBuffer<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim>(b),
          area{ CORE + BORDER + GUARD }
    {
        for( int i=1; i < 27; ++i)
            this->directions = this->directions + Mask(i);
    }

    /**
     * Returns a ParticlesBox for device frame data.
     *
     * @return device frames ParticlesBox
     */
    ParticlesBoxType getParticlesBox()
    {
        return ParticlesBoxType(
            this->superCells.device().data().getDataBox(),
            this->m_deviceHeap->getAllocatorHandle()
        );
    }

    //! only reduces resource access, not memory offset
    auto access_dataPlace( uint32_t area ) const
    {
        auto n = *this;
        n.area = area;
        n.directions = this->directions;
        return n;
    }

    //! only reduces resource access, not memory offset
    auto access_directions( Mask directions ) const
    {
        auto n = *this;
        n.area = this->area;
        n.directions = directions;
        return n;
    }
};

} // namespace particles_buffer

} // namespace pmacc


template<typename T_ParticleDescription, class T_SuperCellSize, typename T_DeviceHeap, unsigned T_dim>
struct redGrapes::trait::BuildProperties<
    pmacc::particles_buffer::HostGuard<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim>
>
{
    template <typename Builder>
    static void build(
        Builder& builder,
        pmacc::particles_buffer::HostGuard<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim> const & buf
    )
    {
        builder.add( buf.superCells.host().data() );//.access_dataPlace( buf.area ).access_directions( buf.exchangeType ) );
    }
};

template<typename T_ParticleDescription, class T_SuperCellSize, typename T_DeviceHeap, unsigned T_dim>
struct redGrapes::trait::BuildProperties<
    pmacc::particles_buffer::DeviceGuard<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim>
>
{
    template <typename Builder>
    static void build(
        Builder& builder,
        pmacc::particles_buffer::DeviceGuard<T_ParticleDescription, T_SuperCellSize, T_DeviceHeap, T_dim> const& buf
    )
    {
        builder.add( buf.superCells.device().data().access_dataPlace( buf.area ).access_directions( buf.directions ) );
    }
};



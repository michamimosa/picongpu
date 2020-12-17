/* Copyright 2013-2020 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch, Benjamin Worpitz, Michael Sippel
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/FieldJ.kernel"
#include "picongpu/fields/currentDeposition/Deposit.hpp"
#include "picongpu/particles/traits/GetCurrentSolver.hpp"
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <pmacc/particles/memory/boxes/ParticlesBox.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/fields/operations/CopyGuardToExchange.hpp>
#include <pmacc/fields/operations/AddExchangeToBorder.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>

#include <boost/mpl/accumulate.hpp>

#include <iostream>
#include <memory>


namespace picongpu
{

using namespace pmacc;

FieldJ::FieldJ( MappingDesc const & cellDescription ) :
    SimulationFieldHelper<MappingDesc>( cellDescription ),
    buffer( cellDescription.getGridLayout( ) ),
    fieldJrecv( nullptr )
{
    const DataSpace<simDim> coreBorderSize = cellDescription.getGridLayout( ).getDataSpaceWithoutGuarding( );

    /* cell margins the current might spread to due to particle shapes */
    using AllSpeciesWithCurrent = typename pmacc::particles::traits::FilterByFlag<
        VectorAllSpecies,
        current<>
    >::type;

    using LowerMarginShapes = bmpl::accumulate<
        AllSpeciesWithCurrent,
        typename pmacc::math::CT::make_Int<simDim, 0>::type,
        pmacc::math::CT::max<bmpl::_1, GetLowerMargin< GetCurrentSolver<bmpl::_2> > >
        >::type;

    using UpperMarginShapes = bmpl::accumulate<
        AllSpeciesWithCurrent,
        typename pmacc::math::CT::make_Int<simDim, 0>::type,
        pmacc::math::CT::max<bmpl::_1, GetUpperMargin< GetCurrentSolver<bmpl::_2> > >
        >::type;

    /* margins are always positive, also for lower margins
     * additional current interpolations and current filters on FieldJ might
     * spread the dependencies on neighboring cells
     *   -> use max(shape,filter) */
    using LowerMargin = pmacc::math::CT::max<
        LowerMarginShapes,
        GetMargin<typename fields::Solver::CurrentInterpolation>::LowerMargin
        >::type;

    using UpperMargin = pmacc::math::CT::max<
        UpperMarginShapes,
        GetMargin<typename fields::Solver::CurrentInterpolation>::UpperMargin
        >::type;

    const DataSpace<simDim> originGuard( LowerMargin( ).toRT( ) );
    const DataSpace<simDim> endGuard( UpperMargin( ).toRT( ) );

    /*go over all directions*/
    for ( uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i )
    {
        DataSpace<simDim> relativMask = Mask::getRelativeDirections<simDim > ( i );
        /*guarding cells depend on direction
         */
        DataSpace<simDim> guardingCells;
        for ( uint32_t d = 0; d < simDim; ++d )
        {
            /*originGuard and endGuard are switch because we send data
             * e.g. from left I get endGuardingCells and from right I originGuardingCells
             */
            switch ( relativMask[d] )
            {
                // receive from negativ side to positiv (end) guarding cells
            case -1: guardingCells[d] = endGuard[d];
                break;
                // receive from positiv side to negativ (origin) guarding cells
            case 1: guardingCells[d] = originGuard[d];
                break;
            case 0: guardingCells[d] = coreBorderSize[d];
                break;
            };

        }
        // std::cout << "ex " << i << " x=" << guardingCells[0] << " y=" << guardingCells[1] << " z=" << guardingCells[2] << std::endl;
        buffer.addExchangeBuffer( i, guardingCells, FIELD_J );
    }

    /* Receive border values in own guard for "receive" communication pattern - necessary for current interpolation/filter */
    const DataSpace<simDim> originRecvGuard( GetMargin<typename fields::Solver::CurrentInterpolation>::LowerMargin( ).toRT( ) );
    const DataSpace<simDim> endRecvGuard( GetMargin<typename fields::Solver::CurrentInterpolation>::UpperMargin( ).toRT( ) );
    if( originRecvGuard != DataSpace<simDim>::create(0) ||
        endRecvGuard != DataSpace<simDim>::create(0) )
    {
        fieldJrecv = std::make_unique< GridBuffer<ValueType, simDim > >(
            buffer.device(),
            cellDescription.getGridLayout( )
        );

        /*go over all directions*/
        for ( uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i )
        {
            DataSpace<simDim> relativMask = Mask::getRelativeDirections<simDim > ( i );
            /* guarding cells depend on direction
             * for negative direction use originGuard else endGuard (relative direction ZERO is ignored)
             * don't switch end and origin because this is a read buffer and no send buffer
             */
            DataSpace<simDim> guardingCells;
            for ( uint32_t d = 0; d < simDim; ++d )
                guardingCells[d] = ( relativMask[d] == -1 ? originRecvGuard[d] : endRecvGuard[d] );
            fieldJrecv->addExchange( GUARD, i, guardingCells, FIELD_JRECV );
        }
    }
}

GridBuffer<FieldJ::ValueType, simDim> &FieldJ::getGridBuffer( )
{
    return buffer;
}

GridLayout<simDim> FieldJ::getGridLayout( )
{
    return cellDescription.getGridLayout( );
}

void FieldJ::communication( )
{
    for (uint32_t i = 1; i < pmacc::traits::NumberOfExchanges<simDim>::value; ++i)
    {
        if ( buffer.hasSendExchange( i ) )
        {
            bashField( i );
            buffer.send( i );
        }

        if ( buffer.hasReceiveExchange( i ) )
        {
            buffer.recv( i );
            insertField( i );
        }
    }

    if( fieldJrecv != nullptr )
        fieldJrecv->communication();
}

void FieldJ::reset( uint32_t )
{
}

void FieldJ::synchronize( )
{
    pmacc::mem::buffer::copy(
        this->host().write(),
        this->device().read()
    );
}

SimulationDataId FieldJ::getUniqueId( )
{
    return getName( );
}

HDINLINE
FieldJ::UnitValueType
FieldJ::getUnit( )
{
    const float_64 UNIT_CURRENT = UNIT_CHARGE / UNIT_TIME / ( UNIT_LENGTH * UNIT_LENGTH );
    return UnitValueType( UNIT_CURRENT, UNIT_CURRENT, UNIT_CURRENT );
}

HINLINE
std::vector<float_64>
FieldJ::getUnitDimension( )
{
    /* L, M, T, I, theta, N, J
    *
    * J is in A/m^2
    *   -> L^-2 * I
    */
    std::vector<float_64> unitDimension( 7, 0.0 );
    unitDimension.at(SIBaseUnits::length) = -2.0;
    unitDimension.at(SIBaseUnits::electricCurrent) =  1.0;

    return unitDimension;
}

std::string
FieldJ::getName( )
{
    return "J";
}

void FieldJ::assign( ValueType value )
{
    pmacc::mem::buffer::fill( this->device(), value );
    //fieldJ.reset(false);
}

template<uint32_t T_area, class T_Species>
void FieldJ::computeCurrent( T_Species & species, uint32_t )
{
    /* tuning parameter to use more workers than cells in a supercell
    * valid domain: 1 <= workerMultiplier
    */
    const int workerMultiplier = 2;

    using FrameType = typename T_Species::FrameType;
    typedef typename pmacc::traits::Resolve<
        typename GetFlagType<FrameType, current<> >::type
    >::type ParticleCurrentSolver;

    using FrameSolver = currentSolver::ComputePerFrame<ParticleCurrentSolver, Velocity, MappingDesc::SuperCellSize>;

    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        typename GetMargin<ParticleCurrentSolver>::LowerMargin,
        typename GetMargin<ParticleCurrentSolver>::UpperMargin
    > BlockArea;

    constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
        pmacc::math::CT::volume< SuperCellSize >::type::value * workerMultiplier
    >::value;

    using Strategy = currentSolver::traits::GetStrategy_t< FrameSolver> ;

    auto const depositionKernel = currentSolver::KernelComputeCurrent<
        numWorkers,
        BlockArea
    >{};

    typename T_Species::ParticlesBoxType pBox = species.getParticlesBuffer( ).device( ).getParticlesBox( );
    FieldJ::DataBoxType jBox = buffer.device( ).data( ).getDataBox( );
    FrameSolver solver( DELTA_T );

    auto const deposit = currentSolver::Deposit< Strategy >{};
    deposit.template execute<
        T_area,
        numWorkers
    >(
        cellDescription,
        depositionKernel,
        solver,
        jBox,
        pBox
    );
}

template<uint32_t T_area, class T_CurrentInterpolation>
void FieldJ::addCurrentToEMF( T_CurrentInterpolation& myCurrentInterpolation )
{
    DataConnector & dc = Environment<>::get().DataConnector();

    Environment<>::task(
        [
            cellDescription = this->cellDescription,
            myCurrentInterpolation
        ]
        (
            auto fieldE,
            auto fieldB,
            auto buffer
        )
        {
            AreaMapping<
                T_area,
                MappingDesc
            > mapper( cellDescription );

            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                pmacc::math::CT::volume< SuperCellSize >::type::value
            >::value;

            PMACC_KERNEL( currentSolver::KernelAddCurrentToEMF< numWorkers >{} )(
                mapper.getGridDim(),
                numWorkers
            )(
                fieldE.getDataBox( ),
                fieldB.getDataBox( ),
                buffer.getDataBox( ),
                myCurrentInterpolation,
                mapper
            );
        },

        TaskProperties::Builder()
            .label("FieldJ::addCurrentToEMF()")
            .scheduling_tags({ SCHED_CUPLA }),

        dc.get< FieldE >( FieldE::getName(), true )->device().data(),
        dc.get< FieldB >( FieldB::getName(), true )->device().data(),
        this->device().data()
    );
}

void FieldJ::bashField( uint32_t exchangeType )
{
    Environment<>::task(
        [ exchangeType ]( auto buffer )
	{
            pmacc::fields::operations::CopyGuardToExchange{ }(
                buffer,
                SuperCellSize{ },
                exchangeType
            );
	},

        TaskProperties::Builder()
            .label("FieldJ::bashField()")
            .scheduling_tags({ SCHED_CUPLA }),

        this->getGridBuffer()
    );
}

void FieldJ::insertField( uint32_t exchangeType )
{
    Environment<>::task(
        [ exchangeType ]( auto buffer )
	{
            pmacc::fields::operations::AddExchangeToBorder{ }(
                buffer,
                SuperCellSize{ },
                exchangeType
            );
	},

        TaskProperties::Builder()
            .label("FieldJ::insertField()")
            .scheduling_tags({ SCHED_CUPLA }),

        this->getGridBuffer()
    );
}

} // namespace picongpu


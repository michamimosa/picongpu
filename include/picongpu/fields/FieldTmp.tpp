/* Copyright 2013-2020 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch, Benjamin Worpitz
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
#include "picongpu/fields/FieldTmp.kernel"
#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/particles/traits/GetInterpolation.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/fields/operations/CopyGuardToExchange.hpp>
#include <pmacc/fields/operations/AddExchangeToBorder.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>

#include <boost/mpl/accumulate.hpp>
#include <string>
#include <memory>


namespace picongpu
{
    using namespace pmacc;

    FieldTmp::FieldTmp(
        MappingDesc const & cellDescription,
        uint32_t slotId
    ) :
        SimulationFieldHelper<MappingDesc>( cellDescription ),
        m_slotId( slotId )
    {
        /* Since this class is instantiated for each temporary field slot,
         * use getNextId( ) directly to get unique tags for each instance.
         * Add SPECIES_FIRSTTAG to avoid collisions with the tags for
         * other fields.
         */
        m_commTagScatter = pmacc::traits::getNextId( ) + SPECIES_FIRSTTAG;
        m_commTagGather = pmacc::traits::getNextId( ) + SPECIES_FIRSTTAG;

        using Buffer = GridBuffer< ValueType, simDim >;
        fieldTmp = std::make_unique< Buffer >( cellDescription.getGridLayout( ) );

        if( fieldTmpSupportGatherCommunication )
            fieldTmpRecv = std::make_unique< Buffer >(
                fieldTmp->device(),
                cellDescription.getGridLayout( )
            );

        /** \todo The exchange has to be resetted and set again regarding the
         *  temporary "Fill-"Functor we want to use.
         *
         *  Problem: buffers don't allow "bigger" exchange during run time.
         *           so let's stay with the maximum guards.
         */
        const DataSpace<simDim> coreBorderSize = cellDescription.getGridLayout( ).getDataSpaceWithoutGuarding( );

        typedef typename pmacc::particles::traits::FilterByFlag
        <
            VectorAllSpecies,
            interpolation<>
        >::type VectorSpeciesWithInterpolation;

        /* ------------------ lower margin  ----------------------------------*/
        typedef bmpl::accumulate<
            VectorSpeciesWithInterpolation,
            typename pmacc::math::CT::make_Int<simDim, 0>::type,
            pmacc::math::CT::max<bmpl::_1, GetLowerMargin< GetInterpolation<bmpl::_2> > >
        >::type SpeciesLowerMargin;

        typedef bmpl::accumulate<
            FieldTmpSolvers,
            typename pmacc::math::CT::make_Int<simDim, 0>::type,
            pmacc::math::CT::max<bmpl::_1, GetLowerMargin< bmpl::_2 > >
        >::type FieldTmpLowerMargin;

        typedef pmacc::math::CT::max<
            SpeciesLowerMargin,
            FieldTmpLowerMargin>::type SpeciesFieldTmpLowerMargin;

        typedef pmacc::math::CT::max<
            GetMargin<fields::Solver, FIELD_B>::LowerMargin,
            GetMargin<fields::Solver, FIELD_E>::LowerMargin>::type
            FieldSolverLowerMargin;

        typedef pmacc::math::CT::max<
            SpeciesFieldTmpLowerMargin,
            FieldSolverLowerMargin>::type LowerMargin;


        /* ------------------ upper margin  -----------------------------------*/

        typedef bmpl::accumulate<
            VectorSpeciesWithInterpolation,
            typename pmacc::math::CT::make_Int<simDim, 0>::type,
            pmacc::math::CT::max<bmpl::_1, GetUpperMargin< GetInterpolation<bmpl::_2> > >
        >::type SpeciesUpperMargin;

        typedef bmpl::accumulate<
            FieldTmpSolvers,
            typename pmacc::math::CT::make_Int<simDim, 0>::type,
            pmacc::math::CT::max<bmpl::_1, GetUpperMargin< bmpl::_2 > >
        >::type FieldTmpUpperMargin;

        typedef pmacc::math::CT::max<
            SpeciesUpperMargin,
            FieldTmpUpperMargin>::type SpeciesFieldTmpUpperMargin;

        typedef pmacc::math::CT::max<
            GetMargin<fields::Solver, FIELD_B>::UpperMargin,
            GetMargin<fields::Solver, FIELD_E>::UpperMargin>::type
            FieldSolverUpperMargin;

        typedef pmacc::math::CT::max<
            SpeciesFieldTmpUpperMargin,
            FieldSolverUpperMargin>::type UpperMargin;

        const DataSpace<simDim> originGuard( LowerMargin( ).toRT( ) );
        const DataSpace<simDim> endGuard( UpperMargin( ).toRT( ) );

        /*go over all directions*/
        for( uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i )
        {
            DataSpace<simDim> relativMask = Mask::getRelativeDirections<simDim > ( i );
            /*guarding cells depend on direction
             */
            DataSpace<simDim> guardingCells;
            for( uint32_t d = 0; d < simDim; ++d )
            {
                /*originGuard and endGuard are switch because we send data
                 * e.g. from left I get endGuardingCells and from right I originGuardingCells
                 */
                switch( relativMask[d] )
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

            fieldTmp->addExchangeBuffer( i, guardingCells, m_commTagScatter );

            if( fieldTmpRecv )
            {
                /* guarding cells depend on direction
                 * for negative direction use originGuard else endGuard (relative direction ZERO is ignored)
                 * don't switch end and origin because this is a read buffer and not send buffer
                 */
                for ( uint32_t d = 0; d < simDim; ++d )
                    guardingCells[d] = ( relativMask[d] == -1 ? originGuard[d] : endGuard[d] );
                fieldTmpRecv->addExchange( GUARD, i, guardingCells, m_commTagGather );
            }
        }

    }

    template<uint32_t AREA, class FrameSolver, class ParticlesClass>
    void FieldTmp::computeValue( ParticlesClass& parClass, uint32_t )
    {
        Environment<>::task(
            [ cellDescription = this->cellDescription ]( auto tmpData, auto parDevice )
            {
                typedef SuperCellDescription<
                    typename MappingDesc::SuperCellSize,
                    typename FrameSolver::LowerMargin,
                    typename FrameSolver::UpperMargin
                > BlockArea;

                StrideMapping<AREA, 3, MappingDesc> mapper( cellDescription );

                FrameSolver solver;
                constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                    pmacc::math::CT::volume< SuperCellSize >::type::value
                >::value;

                do
                {
                    PMACC_KERNEL(
                        KernelComputeSupercells<
                            numWorkers,
                            BlockArea
                        >{ }
                    )(
                        mapper.getGridDim( ),
        		numWorkers
                    )(
                        tmpData.getDataBox(),
                        parDevice.getParticlesBox(),
                        solver,
                        mapper
                    );
                } while( mapper.next( ) );
	    },

            TaskProperties::Builder()
                .label("FieldTmp::computeValue()")
                .scheduling_tags({ SCHED_CUPLA }),

	    this->device().data(),
	    parClass.device()
        );
    }

    SimulationDataId
    FieldTmp::getUniqueId( uint32_t slotId )
    {
        return getName() + std::to_string( slotId );
    }

    SimulationDataId
    FieldTmp::getUniqueId()
    {
        return getUniqueId( m_slotId );
    }

    void FieldTmp::synchronize( )
    {
        pmacc::mem::buffer::copy(
            fieldTmp->host().write(),
            fieldTmp->device().read()
        );
    }

    void FieldTmp::syncToDevice( )
    {
        pmacc::mem::buffer::copy(
            fieldTmp->device().write(),
            fieldTmp->host().read()
        );
    }

    void FieldTmp::communication( )
    {
        for (uint32_t i = 1; i < pmacc::traits::NumberOfExchanges<simDim>::value; ++i)
        {
            if ( fieldTmp->hasSendExchange( i ) )
            {
                bashField( i );
                fieldTmp->send( i );
            }

            if ( fieldTmp->hasReceiveExchange( i ) )
            {
                fieldTmp->recv( i );
                insertField( i );
            }
        }
    }

    void FieldTmp::communicationGather( )
    {
        PMACC_VERIFY_MSG(
            fieldTmpSupportGatherCommunication == true,
            "fieldTmpSupportGatherCommunication in memory.param must be set to true"
        );

        if( fieldTmpRecv != nullptr )
            fieldTmpRecv->communication( );
    }

    void FieldTmp::bashField( uint32_t exchangeType )
    {
        pmacc::fields::operations::CopyGuardToExchange{ }(
            this->getGridBuffer(),
            SuperCellSize{ },
            exchangeType
        );
    }

    void FieldTmp::insertField( uint32_t exchangeType )
    {
        pmacc::fields::operations::AddExchangeToBorder{ }(
            this->getGridBuffer(),
            SuperCellSize{ },
            exchangeType
        );
    }

    GridBuffer<typename FieldTmp::ValueType, simDim> &FieldTmp::getGridBuffer( )
    {
        return *fieldTmp;
    }

    GridLayout< simDim> FieldTmp::getGridLayout( )
    {
        return cellDescription.getGridLayout( );
    }

    void FieldTmp::reset( uint32_t )
    {
        pmacc::mem::buffer::reset( fieldTmp->host(), true );
        pmacc::mem::buffer::reset( fieldTmp->device(), false );
    }

    template<class FrameSolver >
    HDINLINE FieldTmp::UnitValueType
    FieldTmp::getUnit( )
    {
        return FrameSolver().getUnit();
    }

    template<class FrameSolver >
    HINLINE std::vector<float_64>
    FieldTmp::getUnitDimension( )
    {
        return FrameSolver().getUnitDimension();
    }

    std::string
    FieldTmp::getName( )
    {
        return "FieldTmp";
    }

} // namespace picongpu

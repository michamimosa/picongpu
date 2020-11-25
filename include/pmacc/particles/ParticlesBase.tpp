/* Copyright 2013-2020 Heiko Burau, Rene Widera
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

#include "pmacc/Environment.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"

#include "pmacc/fields/SimulationFieldHelper.hpp"
#include "pmacc/mappings/kernel/ExchangeMapping.hpp"

#include "pmacc/particles/memory/boxes/ParticlesBox.hpp"
#include "pmacc/particles/memory/buffers/ParticlesBuffer.hpp"


namespace pmacc
{
    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::deleteGuardParticles(uint32_t exchangeType)
    {
	Environment<>::task(
            [cellDescription, exchangeType] ( auto particlesBufferDev )
	    {
                ExchangeMapping<GUARD, MappingDesc> mapper(cellDescription, exchangeType);

                constexpr uint32_t numWorkers = traits::GetNumWorkers<
                     math::CT::volume< typename FrameType::SuperCellSize >::type::value
                >::value;

                PMACC_KERNEL( KernelDeleteParticles< numWorkers >{ } )(
                    mapper.getGridDim( ),
		    numWorkers
                )(
                    particlesBufferDev->getParticleBox( ),
                    mapper
                );
	    },
	    TaskProperties::Builder()
                .label("ParticlesBase::deleteGuardParticles")
                .schedulingTags({ SCHED_CUPLA }),
	    particlesBuffer.device()
	);
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    template<uint32_t T_area>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::deleteParticlesInArea()
    {
        AreaMapping<T_area, MappingDesc> mapper(this->cellDescription);

        constexpr uint32_t numWorkers = traits::GetNumWorkers<
            math::CT::volume< typename FrameType::SuperCellSize >::type::value
        >::value;

	Environment<>::task(
             []( auto particlesBuffer )
	     {
	        PMACC_KERNEL( KernelDeleteParticles< numWorkers >{ } )(
	            mapper.getGridDim( ),
                    numWorkers
                )(
                    particlesBuffer->getDeviceParticleBox( ),
                    mapper
                );
	     },
             TaskProperties::Builder()
                .label("deleteParticlesInArea")
                .schedulingTags({ SCHED_CUPLA }),
             particlesBuffer.device()
	);

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::reset(uint32_t )
    {
        deleteParticlesInArea<CORE+BORDER+GUARD>();
        particlesBuffer->reset( );
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::copyGuardToExchange( uint32_t exchangeType )
    {
        if( particlesBuffer->hasSendExchange( exchangeType ) )
        {
            particlesBuffer->getSendExchangeStack( exchangeType ).setCurrentSize( 0 );

            Environment<>::task(
                [cellDescription, exchangeType] ( auto particlesBuffer )
                {
                    ExchangeMapping<
                        GUARD,
                        MappingDesc
                    > mapper(
                        cellDescription,
                        exchangeType
                    );

                    constexpr uint32_t numWorkers = traits::GetNumWorkers<
                        math::CT::volume< typename FrameType::SuperCellSize >::type::value
                    >::value;

                    PMACC_KERNEL( KernelCopyGuardToExchange< numWorkers >{ } )(
                        mapper.getGridDim( ),
                        numWorkers
                    )(
                        particlesBuffer->getDeviceParticleBox( ),
                        particlesBuffer->getSendExchangeStack( exchangeType ).getDeviceExchangePushDataBox( ),
                        mapper
                    );
                },
		TaskProperties::Builder()
                    .label("KernelCopyGuardToExchange")
                    .scheduling_tags({ SCHED_CUPLA }),
                particlesBuffer.write()
            );
	}	
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::insertParticles(uint32_t exchangeType)
    {
        if( particlesBuffer->hasReceiveExchange( exchangeType ) )
        {
            size_t numParticles = 0u;
            if( Environment<>::get().isMpiDirectEnabled() )
                numParticles = particlesBuffer.getReceiveExchangeStack( exchangeType ).device().getCurrentSize();
            else
                numParticles = particlesBuffer.getReceiveExchangeStack( exchangeType ).host().getCurrentSize();

            if( numParticles != 0u )
            {
                Environment<>::task(
                    [ cellDescription, exchangeType ] (
                        auto particlesBuffer,
                        auto exchangeStack
                    )
                    {
                        ExchangeMapping<
                            GUARD,
                            MappingDesc
                        > mapper(
                            cellDescription,
                            exchangeType
                        );

                        constexpr uint32_t numWorkers = traits::GetNumWorkers<
                            math::CT::volume< typename FrameType::SuperCellSize >::type::value
                        >::value;

                        PMACC_KERNEL( KernelInsertParticles< numWorkers >{ } )(
                            numParticles,
                            numWorkers
                        )(
                            particlesBuffer->getParticleBox( ),
                            exchangeStack->getPopDataBox( ),
                            mapper
                        );
		    },

                    TaskProperties::Builder()
                        .label("ParticlesBase::insertParticles()")
                        .scheduling_tags({ SCHED_CUPLA }),

                    particlesBuffer.device(),
                    particlesBuffer.getReceiveExchangeStack( exchangeType ).device()
		);
	    }
        }
    }

} //namespace pmacc

#include "pmacc/particles/AsyncCommunicationImpl.hpp"

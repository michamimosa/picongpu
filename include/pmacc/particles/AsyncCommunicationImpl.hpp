/* Copyright 2013-2020 Heiko Burau, Rene Widera, Alexander Grund,
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

#include "pmacc/Environment.hpp"
#include "pmacc/communication/AsyncCommunication.hpp"
#include "pmacc/particles/ParticlesBase.hpp"
#include "pmacc/types.hpp"

#include <boost/type_traits.hpp>

namespace pmacc
{
    /**
     * Trait that should return true if T is a particle species
     */
    template<typename T>
    struct IsParticleSpecies
    {
        enum
        {
            value = boost::is_same<typename T::SimulationDataTag, ParticlesTag>::value
        };
    };

    namespace communication
    {
        template<typename T_Data>
        struct AsyncCommunicationImpl<T_Data, Bool2Type<IsParticleSpecies<T_Data>::value>>
        {
            template<class T_Particles>
            void operator()(T_Particles& particles) const
            {
                typename T_Particles::HandleGuardRegion::HandleExchanged handleExchanged;
                typename T_Particles::HandleGuardRegion::HandleNotExchanged handleNotExchanged;

                size_t n_exchanges = traits::NumberOfExchanges<T_Particles::Dim>::value;

                for(size_t i = 1; i < n_exchanges; ++i)
                {
                    if(particles.getParticlesBuffer().hasSendExchange(i))
                        handleExchanged.handleOutgoing(particles, i);
                    else
                        handleNotExchanged.handleOutgoing(particles, i);
                }

                for(size_t i = 1; i < n_exchanges; ++i)
                {
                    if(particles.getParticlesBuffer().hasReceiveExchange(i))
                        handleExchanged.handleIncoming(particles, i);
                    else
                        handleNotExchanged.handleIncoming(particles, i);
                }

                particles.fillBorderGaps();
            }
        };

    } // namespace communication
} // namespace pmacc

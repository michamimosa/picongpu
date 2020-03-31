/* Copyright 2013-2019 Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
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

#include "pmacc/types.hpp"
#include "pmacc/dimensions/DataSpace.hpp"

#include <mpi.h>

namespace pmacc
{

/*! Interface for communication
 */
class ICommunicator
{
public:

    /*! returns available communication partners
     *
     * @return a mask with neighbors, e.g. if there is a right neighbor result.isSet(RIGHT) returns true
     */
    virtual const Mask& getCommunicationMask() const=0;

    /*! moves all GPUs from top to bottom (y-coordinate)
     *
     * @return true if the position of gpu is switched to the end, else false
     */
    virtual bool slide() = 0;

    /*! slides multiple times
     *
     * @param[in] numSlides number of slides
     * @return true if the position of gpu is switched to the end, else false
     */
    virtual bool setStateAfterSlides(size_t numSlides) = 0;

    /*! start a sending task
     *
     * @param ex                direction to send (enum ExchangeType)
     * @param send_data         pointer to data; should have at least send_data_count bytes
     * @param send_data_count   message size in bytes to sent
     * @param tag               user-defined tag; only message with the same tag can be exchanged (i.e. startSend and startReceive must use the same tag)
     */
    virtual void send(
        uint32_t ex,
        char const * send_data,
        size_t send_data_count,
        uint32_t tag
    ) = 0;

    /*! start a receiving task
     *
     * If recv_data_max is less then send_data_count (on other host) multiple startReceive are needed!
     *
     * @param ex                direction to send (enum ExchangeType)
     * @param recv_data         pointer to data; should have at least recv_data_max bytes
     * @param recv_data_max     maximum message size in bytes to receive
     * @param tag               user-defined tag; only message with the same tag can be exchanged (i.e. send and receive must use the same tag)
     *
     * @return number of received bytes
     */
    virtual size_t recv(
        uint32_t ex,
        char * recv_data,
        size_t recv_data_max,
        uint32_t tag
    ) = 0;

    virtual int getRank()=0;

    /*! Return which of the three directions are periodic
     *
     * @return for each direction a false (0) or true(1) value
     */
    virtual DataSpace<DIM3> getPeriodic() const = 0;
};

} //namespace pmacc


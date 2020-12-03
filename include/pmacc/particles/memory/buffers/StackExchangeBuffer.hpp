/* Copyright 2013-2020 Felix Schmitt, Rene Widera, Benjamin Worpitz,
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

#include "pmacc/assert.hpp"
#include "pmacc/memory/buffers/gridBuffer/Exchange.hpp"
#include "pmacc/particles/memory/boxes/ExchangePopDataBox.hpp"
#include "pmacc/particles/memory/boxes/ExchangePushDataBox.hpp"

namespace pmacc
{

namespace stack_exchange_buffer
{
template <class FRAME, class FRAMEINDEX, unsigned DIM>
struct HostGuard;

template <class FRAME, class FRAMEINDEX, unsigned DIM>
struct DeviceGuard;
}

    /**
     * Can be used for creating several DataBox types from an Exchange.
     *
     * @tparam FRAME frame datatype
     */
    template <class FRAME, class FRAMEINDEX, unsigned DIM>
    class StackExchangeBuffer
    {
    public:
        /**
         * Create a stack from any ExchangeBuffer<FRAME,DIM>.
         *
         * If the stack's internal GridBuffer has no sizeOnDevice, no device
         * querys are allowed.
         *
         * @param stack Exchange
         */
        StackExchangeBuffer(
            mem::ExchangeBuffer< FRAME, DIM1, mem::grid_buffer::data::Access > stack,
            mem::ExchangeBuffer< FRAMEINDEX, DIM1, mem::grid_buffer::data::Access > stackIndexer
        ) :
            stack(stack),
            stackIndexer(stackIndexer)
        {}

        void setCurrentSize(const size_t size)
        {
            stack.getHostBuffer().size().set(size);
            stack.getDeviceBuffer().size().set(size);
            stackIndexer.getHostBuffer().size().set(size);
            stackIndexer.getDeviceBuffer().size().set(size);
        }

        size_t getMaxParticlesCount()
        {
            if (Environment<>::get().isMpiDirectEnabled())
                return stack.getDeviceBuffer()
                    .getDataSpace()
                    .productOfComponents();
            else
                return stack.getHostBuffer()
                    .getDataSpace()
                    .productOfComponents();
        }

        auto host()
        {
            return stack_exchange_buffer::HostGuard<FRAME, FRAMEINDEX, DIM>(*this);
        }

        auto device()
        {
            return stack_exchange_buffer::DeviceGuard<FRAME, FRAMEINDEX, DIM>(*this);
        }

    protected:
        auto getExchangeBuffer()
        {
            return stack;
        }

        mem::ExchangeBuffer< FRAME, DIM1, mem::grid_buffer::data::Access > stack;
        mem::ExchangeBuffer< FRAMEINDEX, DIM1, mem::grid_buffer::data::Access > stackIndexer;
    };

    namespace stack_exchange_buffer
    {
        /*
         * Acces Guard that only allows access to host side databoxes
         */
        template <class FRAME, class FRAMEINDEX, unsigned DIM>
        struct HostGuard : private StackExchangeBuffer<FRAME, FRAMEINDEX, DIM>
        {
            friend class redGrapes::trait::BuildProperties< HostGuard >;

            size_t getCurrentSize()
            {
                if (Environment<>::get().isMpiDirectEnabled())
                    return this->stackIndexer.getDeviceBuffer().size().get();
                else
                    return this->stackIndexer.getHostBuffer().size().get();
            }

            size_t getParticlesCurrentSize()
            {
                if (Environment<>::get().isMpiDirectEnabled())
                    return this->stack.getDeviceBuffer().size().get();
                else
                    return this->stack.getHostBuffer().size().get();
            }

            /**
             * Returns a PushDataBox for the internal HostBuffer.
             *
             * @return PushDataBox for host buffer
             */
            ExchangePushDataBox<vint_t, FRAME, DIM>
            getPushDataBox()
            {
                return ExchangePushDataBox<vint_t, FRAME, DIM>(
                    this->stack.getHostBuffer().data().getBasePointer(),
                    this->stack.getHostBuffer().size().get_host_pointer(),
                    this->stack.getHostBuffer().getDataSpace().productOfComponents(),
                    PushDataBox<vint_t, FRAMEINDEX>(
                        this->stackIndexer.getHostBuffer().data().getBasePointer(),
                        this->stackIndexer.getHostBuffer()
                            .size()
                            .get_host_pointer()));
            }

            /**
             * Returns a PopDataBox for the internal HostBuffer.
             *
             * @return PopDataBox for host buffer
             */
            ExchangePopDataBox<vint_t, FRAME, DIM> getPopDataBox()
            {
                return ExchangePopDataBox<vint_t, FRAME, DIM>(
                    this->stack.getHostBuffer().data().getDataBox(),
                    this->stackIndexer.getHostBuffer().data().getDataBox());
            }
        };

        /*
         * Acces Guard that only allows access to device side databoxes
         */
        template <class FRAME, class FRAMEINDEX, unsigned DIM>
        struct DeviceGuard : private StackExchangeBuffer<FRAME, FRAMEINDEX, DIM>
        {
            friend class redGrapes::trait::BuildProperties< DeviceGuard >;

            size_t getCurrentSize()
            {
                return this->stackIndexer.getDeviceBuffer().size().get();
            }

            size_t getParticlesCurrentSize()
            {
                return this->stack.getDeviceBuffer().size().get();
            }

            /**
             * Returns a PushDataBox for the internal DeviceBuffer.
             *
             * @return PushDataBox for device buffer
             */
            ExchangePushDataBox<vint_t, FRAME, DIM> getPushDataBox()
            {
                PMACC_ASSERT(this->stack.getDeviceBuffer().size().is_on_device());
                PMACC_ASSERT(this->stackIndexer.getDeviceBuffer().size().is_on_device());

                return ExchangePushDataBox<vint_t, FRAME, DIM>(
                    this->stack.getDeviceBuffer().data().getBasePointer(),
                    (vint_t*) this->stack.getDeviceBuffer()
                        .size()
                        .get_device_pointer(),
                    this->stack.getDeviceBuffer()
                        .data()
                        .getDataSpace()
                        .productOfComponents(),
                    PushDataBox<vint_t, FRAMEINDEX>(
                        this->stackIndexer.getDeviceBuffer().data().getBasePointer(),
                        (vint_t*) this->stackIndexer.getDeviceBuffer()
                            .size()
                            .get_device_pointer()));
            }

            /**
             * Returns a PopDataBox for the internal DeviceBuffer.
             *
             * @return PopDataBox for device buffer
             */
            ExchangePopDataBox<vint_t, FRAME, DIM>
            getPopDataBox()
            {
                return ExchangePopDataBox<vint_t, FRAME, DIM>(
                    this->stack.getDeviceBuffer().data().getDataBox(),
                    this->stackIndexer.getDeviceBuffer().data().getDataBox());
            }
        };

    } // namespace stack_exchange_buffer

} // namespace pmacc

template <class FRAME, class FRAMEINDEX, unsigned DIM>
struct redGrapes::trait::BuildProperties<
    pmacc::StackExchangeBuffer<FRAME, FRAMEINDEX, DIM>
>
{
    template <typename Builder>
    static void build(
        Builder& builder,
        pmacc::StackExchangeBuffer<FRAME, FRAMEINDEX, DIM> const & buf
    )
    {
        builder.add( buf.stack );
        builder.add( buf.stackIndexer );
    }
};

template <class FRAME, class FRAMEINDEX, unsigned DIM>
struct redGrapes::trait::BuildProperties<
    pmacc::stack_exchange_buffer::HostGuard<FRAME, FRAMEINDEX, DIM>
>
{
    template <typename Builder>
    static void build(
        Builder& builder,
        pmacc::stack_exchange_buffer::DeviceGuard<FRAME, FRAMEINDEX, DIM> const & buf
    )
    {
        builder.add( buf.stack.host() );
        builder.add( buf.stackIndexer.host() );
    }
};

template <class FRAME, class FRAMEINDEX, unsigned DIM>
struct redGrapes::trait::BuildProperties<
    pmacc::stack_exchange_buffer::DeviceGuard<FRAME, FRAMEINDEX, DIM>
>
{
    template <typename Builder>
    static void build(
        Builder& builder,
        pmacc::stack_exchange_buffer::DeviceGuard<FRAME, FRAMEINDEX, DIM> const & buf
    )
    {
        builder.add( buf.stack.device() );
        builder.add( buf.stackIndexer.device() );
    }
};


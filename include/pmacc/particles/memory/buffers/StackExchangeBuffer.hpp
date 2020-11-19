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
        StackExchangeBuffer(mem::ExchangeBuffer<FRAME, DIM1>& stack,
                            mem::ExchangeBuffer<FRAMEINDEX, DIM1>& stackIndexer)
            : stack(stack)
            , stackIndexer(stackIndexer)
        {
        }

        void setCurrentSize(const size_t size)
        {
            stack.getHostBuffer().size().set(size);
            stack.getDeviceBuffer().size().set(size);
            stackIndexer.getHostBuffer().size().set(size);
            stackIndexer.getDeviceBuffer().size().set(size);
        }

        size_t getHostCurrentSize()
        {
            if (Environment<>::get().isMpiDirectEnabled())
                return stackIndexer.getDeviceBuffer().size().get();
            else
                return stackIndexer.getDeviceBuffer().size().get();
        }

        size_t getHostParticlesCurrentSize()
        {
            if (Environment<>::get().isMpiDirectEnabled())
                return stack.getDeviceBuffer().size().get();
            else
                return stack.getHostBuffer().size().get();
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

        size_t getHostParticlesCurrentSize()
        {
            if (Environment<>::get().isMpiDirectEnabled())
                return stack.getDeviceBuffer().size().get();
            else
                return stack.getHostBuffer().size().get();
        }

        auto host()
        {
            return stack_exchange_buffer::HostGuard(*this);
        }

        auto device()
        {
            return stack_exchange_buffer::DeviceGuard(*this);
        }

    private:
        mem::ExchangeBuffer<FRAME, DIM1>& getExchangeBuffer()
        {
            return stack;
        }

        mem::ExchangeBuffer<FRAME, DIM1>& stack;
        mem::ExchangeBuffer<FRAMEINDEX, DIM1>& stackIndexer;
    };


    namespace stack_exchange_buffer
    {
        /*
         * Acces Guard that only allows access to host side databoxes
         */
        struct HostGuard : StackExchangeBuffer
        {
            /**
             * Returns a PushDataBox for the internal HostBuffer.
             *
             * @return PushDataBox for host buffer
             */
            ExchangePushDataBox<vint_t, FRAME, DIM>
            getHostExchangePushDataBox()
            {
                return ExchangePushDataBox<vint_t, FRAME, DIM>(
                    stack.getHostBuffer().data().getBasePointer(),
                    stack.getHostBuffer().size().get_host_pointer(),
                    stack.getHostBuffer().getDataSpace().productOfComponents(),
                    PushDataBox<vint_t, FRAMEINDEX>(
                        stackIndexer.getHostBuffer().data().getBasePointer(),
                        stackIndexer.getHostBuffer()
                            .size()
                            .get_host_pointer()));
            }

            /**
             * Returns a PopDataBox for the internal HostBuffer.
             *
             * @return PopDataBox for host buffer
             */
            ExchangePopDataBox<vint_t, FRAME, DIM> getHostExchangePopDataBox()
            {
                return ExchangePopDataBox<vint_t, FRAME, DIM>(
                    stack.getHostBuffer().data().getDataBox(),
                    stackIndexer.getHostBuffer().data().getDataBox());
            }
        }

        /*
         * Acces Guard that only allows access to device side databoxes
         */
        struct DeviceGuard : StackExchangeBuffer
        {
            size_t getParticlesCurrentSize()
            {
                return stack.getDeviceBuffer().size().get();
            }

            size_t getCurrentSize()
            {
                return stackIndexer.getDeviceBuffer().size().get();
            }

            size_t getParticlesCurrentSize()
            {
                return stack.getDeviceBuffer().size().get();
            }

            /**
             * Returns a PushDataBox for the internal DeviceBuffer.
             *
             * @return PushDataBox for device buffer
             */
            ExchangePushDataBox<vint_t, FRAME, DIM> getExchangePushDataBox()
            {
                PMACC_ASSERT(stack.getDeviceBuffer().size().is_on_device());
                PMACC_ASSERT(stackIndexer.getDeviceBuffer().size().is_on_device());

                return ExchangePushDataBox<vint_t, FRAME, DIM>(
                    stack.getDeviceBuffer().data().getBasePointer(),
                    (vint_t*) stack.getDeviceBuffer()
                        .size()
                        .get_device_pointer(),
                    stack.getDeviceBuffer()
                        .data()
                        .getDataSpace()
                        .productOfComponents(),
                    PushDataBox<vint_t, FRAMEINDEX>(
                        stackIndexer.getDeviceBuffer().data().getBasePointer(),
                        (vint_t*) stackIndexer.getDeviceBuffer()
                            .size()
                            .get_device_pointer()));
            }

            /**
             * Returns a PopDataBox for the internal DeviceBuffer.
             *
             * @return PopDataBox for device buffer
             */
            ExchangePopDataBox<vint_t, FRAME, DIM>
            getDeviceExchangePopDataBox()
            {
                return ExchangePopDataBox<vint_t, FRAME, DIM>(
                    stack.getDeviceBuffer().data().getDataBox(),
                    stackIndexer.getDeviceBuffer().data().getDataBox());
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
        pmacc::StackExchangeBuffer<FRAME, FRAMEINDEX, DIM> const& buf
    )
    {
        builder.add(buf.stack.write());
        builder.add(buf.stackIndexer.write());
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
        pmacc::stack_exchange_buffer::DeviceGuard<FRAME, FRAMEINDEX, DIM> const& buf
    )
    {
        builder.add(buf.stack.device().write());
        builder.add(buf.stackIndexer.device().write());
    }
};


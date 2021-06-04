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
        template<class FRAME, class FRAMEINDEX, unsigned DIM>
        struct HostGuard;

        template<class FRAME, class FRAMEINDEX, unsigned DIM>
        struct DeviceGuard;
    } // namespace stack_exchange_buffer

    /**
     * Can be used for creating several DataBox types from an Exchange.
     *
     * @tparam FRAME frame datatype
     */
    template<class FRAME, class FRAMEINDEX, unsigned DIM>
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
            mem::ExchangeBuffer<FRAME, DIM1, mem::grid_buffer::data::Access> const& stack,
            mem::ExchangeBuffer<FRAMEINDEX, DIM1, mem::grid_buffer::data::Access> const& stackIndexer)
            : stack(stack)
            , stackIndexer(stackIndexer)
        {
        }

        void setCurrentSize(const size_t size)
        {
            if(auto host = stack.host())
                host->size().set(size);
            stack.device().size().set(size);

            if(auto host = stackIndexer.host())
                host->size().set(size);
            stackIndexer.device().size().set(size);
        }

        size_t getMaxParticlesCount()
        {
            if(Environment<>::get().isMpiDirectEnabled())
                return stack.device().getDataSpace().productOfComponents();
            else
                return stack.host()->getDataSpace().productOfComponents();
        }

        auto host() const
        {
            return stack_exchange_buffer::HostGuard<FRAME, FRAMEINDEX, DIM>(*this);
        }

        auto device() const
        {
            return stack_exchange_buffer::DeviceGuard<FRAME, FRAMEINDEX, DIM>(*this);
        }

    protected:
        auto getExchangeBuffer()
        {
            return stack;
        }

        mem::ExchangeBuffer<FRAME, DIM1, mem::grid_buffer::data::Access> stack;
        mem::ExchangeBuffer<FRAMEINDEX, DIM1, mem::grid_buffer::data::Access> stackIndexer;
    };

    namespace stack_exchange_buffer
    {
        /*
         * Acces Guard that only allows access to host side databoxes
         */
        template<class FRAME, class FRAMEINDEX, unsigned DIM>
        struct HostGuard : private StackExchangeBuffer<FRAME, FRAMEINDEX, DIM>
        {
            friend class redGrapes::trait::BuildProperties<HostGuard>;

            HostGuard(StackExchangeBuffer<FRAME, FRAMEINDEX, DIM> const& b)
                : StackExchangeBuffer<FRAME, FRAMEINDEX, DIM>(b)
            {
            }

            size_t getCurrentSize()
            {
                if(Environment<>::get().isMpiDirectEnabled())
                    return this->stackIndexer.device().size().get();
                else
                    return this->stackIndexer.host()->size().get();
            }

            size_t getParticlesCurrentSize()
            {
                if(Environment<>::get().isMpiDirectEnabled())
                    return this->stack.device().size().get();
                else
                    return this->stack.host()->size().get();
            }

            /**
             * Returns a PushDataBox for the internal HostBuffer.
             *
             * @return PushDataBox for host buffer
             */
            ExchangePushDataBox<vint_t, FRAME, DIM> getPushDataBox()
            {
                return ExchangePushDataBox<vint_t, FRAME, DIM>(
                    this->stack.host()->data().getBasePointer(),
                    this->stack.host()->size().get_host_pointer(),
                    this->stack.host()->getDataSpace().productOfComponents(),
                    PushDataBox<vint_t, FRAMEINDEX>(
                        this->stackIndexer.host()->data().getBasePointer(),
                        this->stackIndexer.host()->size().get_host_pointer()));
            }

            /**
             * Returns a PopDataBox for the internal HostBuffer.
             *
             * @return PopDataBox for host buffer
             */
            ExchangePopDataBox<vint_t, FRAME, DIM> getPopDataBox()
            {
                return ExchangePopDataBox<vint_t, FRAME, DIM>(
                    this->stack.host()->data().getDataBox(),
                    this->stackIndexer.host()->data().getDataBox());
            }
        };

        /*
         * Acces Guard that only allows access to device side databoxes
         */
        template<class FRAME, class FRAMEINDEX, unsigned DIM>
        struct DeviceGuard : private StackExchangeBuffer<FRAME, FRAMEINDEX, DIM>
        {
            friend class redGrapes::trait::BuildProperties<DeviceGuard>;

            DeviceGuard(StackExchangeBuffer<FRAME, FRAMEINDEX, DIM> const& b)
                : StackExchangeBuffer<FRAME, FRAMEINDEX, DIM>(b)
            {
            }

            size_t getCurrentSize()
            {
                return this->stackIndexer.device().size().get();
            }

            size_t getParticlesCurrentSize()
            {
                return this->stack.device().size().get();
            }

            /**
             * Returns a PushDataBox for the internal DeviceBuffer.
             *
             * @return PushDataBox for device buffer
             */
            ExchangePushDataBox<vint_t, FRAME, DIM> getPushDataBox()
            {
                PMACC_ASSERT(this->stack.device().size().is_on_device());
                PMACC_ASSERT(this->stackIndexer.device().size().is_on_device());

                return ExchangePushDataBox<vint_t, FRAME, DIM>(
                    this->stack.device().data().getBasePointer(),
                    (vint_t*) this->stack.device().size().get_device_pointer(),
                    this->stack.device().getDataSpace().productOfComponents(),
                    PushDataBox<vint_t, FRAMEINDEX>(
                        this->stackIndexer.device().data().getBasePointer(),
                        (vint_t*) this->stackIndexer.device().size().get_device_pointer()));
            }

            /**
             * Returns a PopDataBox for the internal DeviceBuffer.
             *
             * @return PopDataBox for device buffer
             */
            ExchangePopDataBox<vint_t, FRAME, DIM> getPopDataBox()
            {
                return ExchangePopDataBox<vint_t, FRAME, DIM>(
                    this->stack.device().data().getDataBox(),
                    this->stackIndexer.device().data().getDataBox());
            }
        };

    } // namespace stack_exchange_buffer

} // namespace pmacc

template<class FRAME, class FRAMEINDEX, unsigned DIM>
struct redGrapes::trait::BuildProperties<pmacc::StackExchangeBuffer<FRAME, FRAMEINDEX, DIM>>
{
    template<typename Builder>
    static void build(Builder& builder, pmacc::StackExchangeBuffer<FRAME, FRAMEINDEX, DIM> const& buf)
    {
        builder.add(buf.stack);
        builder.add(buf.stackIndexer);
    }
};

template<class FRAME, class FRAMEINDEX, unsigned DIM>
struct redGrapes::trait::BuildProperties<pmacc::stack_exchange_buffer::HostGuard<FRAME, FRAMEINDEX, DIM>>
{
    template<typename Builder>
    static void build(Builder& builder, pmacc::stack_exchange_buffer::DeviceGuard<FRAME, FRAMEINDEX, DIM> const& buf)
    {
        builder.add(buf.stack.host());
        builder.add(buf.stackIndexer.host());
    }
};

template<class FRAME, class FRAMEINDEX, unsigned DIM>
struct redGrapes::trait::BuildProperties<pmacc::stack_exchange_buffer::DeviceGuard<FRAME, FRAMEINDEX, DIM>>
{
    template<typename Builder>
    static void build(Builder& builder, pmacc::stack_exchange_buffer::DeviceGuard<FRAME, FRAMEINDEX, DIM> const& buf)
    {
        builder.add(buf.stack.device());
        builder.add(buf.stackIndexer.device());
    }
};

